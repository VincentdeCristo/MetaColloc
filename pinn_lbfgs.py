"""
Baseline: Physics-Informed Neural Network (PINN) with L-BFGS
Fair comparison with MetaColloc:
- Same PDEs (6)
- Same seeds (42–46)
- Same interior / boundary / test sizes
- Same precision (float64)
- Same statistics: MSE + 95% CI
"""
from dataclasses import dataclass
from typing import Callable
import sympy as sym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bootstrap
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# PDE registry (Linear & Non-Linear)
# ================================================================
@dataclass
class PDESpec:
    name: str
    u_fn: Callable
    f_fn: Callable


def _build_pde_registry() -> dict:
    x, y = sym.symbols("x y") # Note: for KdV and Sine-Gordon, 'y' acts as time 't'
    registry = {}

    # 1) Poisson
    u_sym = sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y) + sym.exp(-x - y)
    f_sym = -(sym.diff(u_sym, x, 2) + sym.diff(u_sym, y, 2))
    registry["Poisson"] = PDESpec(
        name="Poisson",
        u_fn=sym.lambdify((x, y), u_sym, "numpy"),
        f_fn=sym.lambdify((x, y), f_sym, "numpy"),
    )

    # 2) Helmholtz (Extreme High-Freq, k=64π)
    k = 64 * sym.pi
    kxy = k / sym.sqrt(2)
    u_h = sym.sin(kxy * x) * sym.cos(kxy * y) + sym.exp(-x - y)
    f_h = -(sym.diff(u_h, x, 2) + sym.diff(u_h, y, 2)) - k**2 * u_h
    registry["Helmholtz"] = PDESpec(
        name="Helmholtz",
        u_fn=sym.lambdify((x, y), u_h, "numpy"),
        f_fn=sym.lambdify((x, y), f_h, "numpy"),
    )

    # 3) Variable coefficient
    a_sym = 2 + sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
    u_v = sym.sin(sym.pi * x) * sym.sin(sym.pi * y) + sym.exp(-x - y)
    f_v = -(sym.diff(a_sym * sym.diff(u_v, x), x) + sym.diff(a_sym * sym.diff(u_v, y), y))
    registry["VarCoeff"] = PDESpec(
        name="VarCoeff",
        u_fn=sym.lambdify((x, y), u_v, "numpy"),
        f_fn=sym.lambdify((x, y), f_v, "numpy"),
    )

    # 4) HighFreq Poisson – sin(8πx)sin(8πy) + exp(-xy)
    u_hf = sym.sin(8 * sym.pi * x) * sym.sin(8 * sym.pi * y) + sym.exp(-x * y)
    f_hf = -(sym.diff(u_hf, x, 2) + sym.diff(u_hf, y, 2))
    registry["HighFreq"] = PDESpec(
        name="HighFreq",
        u_fn=sym.lambdify((x, y), u_hf, "numpy"),
        f_fn=sym.lambdify((x, y), f_hf, "numpy"),
    )

    # 5) Sine-Gordon (Non-linear)
    u_sg = sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
    f_sg = sym.diff(u_sg, y, 2) - sym.diff(u_sg, x, 2) + sym.sin(u_sg)
    registry["SineGordon"] = PDESpec(
        name="SineGordon",
        u_fn=sym.lambdify((x, y), u_sg, "numpy"),
        f_fn=sym.lambdify((x, y), f_sg, "numpy"),
    )

    # 6) KdV (Non-linear, 3rd-order derivative)
    u_kdv = sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
    f_kdv = sym.diff(u_kdv, y, 1) + 6 * u_kdv * sym.diff(u_kdv, x, 1) + sym.diff(u_kdv, x, 3)
    registry["KdV"] = PDESpec(
        name="KdV",
        u_fn=sym.lambdify((x, y), u_kdv, "numpy"),
        f_fn=sym.lambdify((x, y), f_kdv, "numpy"),
    )

    return registry


PDE_REGISTRY = _build_pde_registry()
PDE_NAMES = list(PDE_REGISTRY.keys())
print(f"Registered PDEs: {PDE_NAMES}")

_HELMHOLTZ_K2 = (64 * np.pi) ** 2
_a_varcoeff  = lambda x, y:  2 + np.sin(np.pi * x) * np.cos(np.pi * y)
_ax_varcoeff = lambda x, y:  np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
_ay_varcoeff = lambda x, y: -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)

# ================================================================
# Sampling
# ================================================================
def sample_interior(n):
    return np.random.rand(n, 2)

def sample_boundary(Q):
    t = np.linspace(0, 1, Q)
    return np.vstack([
        np.stack([t, np.zeros_like(t)], 1),
        np.stack([t, np.ones_like(t)], 1),
        np.stack([np.zeros_like(t), t], 1),
        np.stack([np.ones_like(t), t], 1),
    ])

# ================================================================
# PINN Model
# ================================================================
class PINN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.fc1 = nn.Linear(d_in, 2 * d_hidden)
        self.fc2 = nn.Linear(2 * d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        return self.out(x)

# ================================================================
# PDE residuals
# ================================================================
def pde_residual(model, x, pde_name, f_fn):
    x.requires_grad_(True)
    u = model(x)

    grads = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grads[:, 0:1], grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    x_np, y_np = x[:, 0].detach().cpu().numpy(), x[:, 1].detach().cpu().numpy()
    f = torch.tensor(f_fn(x_np, y_np)).reshape(-1, 1).to(device)

    if pde_name in ["Poisson", "HighFreq"]:
        return -(u_xx + u_yy) - f

    if pde_name == "Helmholtz":
        return -(u_xx + u_yy) - _HELMHOLTZ_K2 * u - f

    if pde_name == "VarCoeff":
        a  = torch.tensor(_a_varcoeff(x_np, y_np)).reshape(-1,1).to(device)
        ax = torch.tensor(_ax_varcoeff(x_np, y_np)).reshape(-1,1).to(device)
        ay = torch.tensor(_ay_varcoeff(x_np, y_np)).reshape(-1,1).to(device)
        return -(a*(u_xx+u_yy) + ax*u_x + ay*u_y) - f

    if pde_name == "SineGordon":
        return u_yy - u_xx + torch.sin(u) - f

    if pde_name == "KdV":
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0][:,0:1]
        return u_y + 6*u*u_x + u_xxx - f

    raise ValueError(pde_name)

# ================================================================
# Solve one PDE
# ================================================================
def solve_pinn(pde_name, width, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = PINN(d_in=2, d_hidden=width).to(device)

    pts = torch.tensor(sample_interior(2000), device=device)
    bd  = torch.tensor(sample_boundary(300), device=device)

    u_bd_true = torch.tensor(
        PDE_REGISTRY[pde_name].u_fn(
            bd[:,0].cpu(), bd[:,1].cpu()
        )
    ).reshape(-1,1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(1000):
        optimizer.zero_grad()
        r = pde_residual(model, pts, pde_name, PDE_REGISTRY[pde_name].f_fn)
        loss = (r**2).mean() + ((model(bd)-u_bd_true)**2).mean()
        loss.backward()
        optimizer.step()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=500,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        r = pde_residual(model, pts, pde_name, PDE_REGISTRY[pde_name].f_fn)
        loss = (r**2).mean() + ((model(bd)-u_bd_true)**2).mean()
        loss.backward()
        return loss

    optimizer.step(closure)

    xt = torch.tensor(sample_interior(10000), device=device)
    with torch.no_grad():
        u_pred = model(xt).cpu().numpy()

    u_true = PDE_REGISTRY[pde_name].u_fn(
        xt[:,0].cpu().numpy(), xt[:,1].cpu().numpy()
    ).reshape(-1,1)

    return float(np.mean((u_pred - u_true)**2))

# ================================================================
# Main experiment loop
# ================================================================
D_HIDDENS = [128, 256, 512, 1024]
SEEDS = range(42, 47)

results = {pde: {h: [] for h in D_HIDDENS} for pde in PDE_NAMES}
solve_times = {pde: {h: [] for h in D_HIDDENS} for pde in PDE_NAMES}

for pde in PDE_NAMES:
    for h in D_HIDDENS:
        for seed in SEEDS:
            t_solve_start = time.time()
            mse = solve_pinn(pde, h, seed)
            solve_times[pde][h].append(time.time() - t_solve_start)
            rmse = np.sqrt(mse)
            results[pde][h].append(rmse)
            print(f"[PINN] {pde} H={h} seed={seed} RMSE={rmse:.2e} ")

# ================================================================
# Statistics
# ================================================================
print("\n=== PINN + L-BFGS Summary ===")
for pde in PDE_NAMES:
    print(f"\n[{pde}]")
    for h in D_HIDDENS:
        data = np.array(results[pde][h])
        res = bootstrap((data,), np.mean, confidence_level=0.95,
                        method="percentile", n_resamples=10_000, random_state=0)
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        ci_width = hi / 2 - lo / 2
        m = data.mean()
        print(
            f"  H={h:4d}  mean={m:.4e} ± {ci_width:.4e}  "
        )

for pde_name in PDE_NAMES:
    print(f"  [{pde_name}]")
    for h in D_HIDDENS:
        data = np.array(solve_times[pde][h])
        res = bootstrap((data,), np.mean, confidence_level=0.95,
                        method="percentile", n_resamples=10_000, random_state=0)
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        ci_width = hi / 2 - lo / 2
        m = data.mean()
        print(f"    H={h:4d}  Time = {m:.3f}s  ± {ci_width:.3f}s")