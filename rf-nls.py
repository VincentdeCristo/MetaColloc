"""
Baseline: Random Feature Model (RF-NLS) for PDEs
Paper: Solving Partial Differential Equations with Random Feature Models (arXiv:2501.00288v2)

Fair comparison with MetaColloc:
- Same PDEs (6)
- Same seeds (42–46)
- Same interior / boundary / test sizes
- Same precision (float64)
- Same statistics: MSE + 95% CI

Key implementations of Random Feature Models:
1. Frozen Hidden Layer: The frequencies (W) are sampled from N(0, sigma^2) and biases (b) 
   from U[-pi, pi]. They are NOT updated during training.
2. Cosine Feature Map: \phi(x) = 1/sqrt(N) * cos(x W + b).
3. Trainable Coefficients: Only the final linear layer (c) is optimized.
4. NLS Optimization: Utilizes Adam followed by L-BFGS on the tiny parameter space (c) 
   to strictly replicate the "RF-NLS" results reported in the paper.
"""
from dataclasses import dataclass
from typing import Callable
import sympy as sym
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# PDE registry (Linear & Non-Linear) - Exactly as Baseline
# ================================================================
@dataclass
class PDESpec:
    name: str
    u_fn: Callable
    f_fn: Callable

def _build_pde_registry() -> dict:
    x, y = sym.symbols("x y") 
    registry = {}

    # 1) Poisson
    u_sym = sym.sin(2 * sym.pi * x) * sym.sin(2 * sym.pi * y) + sym.exp(-x - y)
    f_sym = -(sym.diff(u_sym, x, 2) + sym.diff(u_sym, y, 2))
    registry["Poisson"] = PDESpec("Poisson", sym.lambdify((x, y), u_sym, "numpy"), sym.lambdify((x, y), f_sym, "numpy"))

    # 2) Helmholtz (Extreme High-Freq, k=64π)
    k = 64 * sym.pi
    kxy = k / sym.sqrt(2)
    u_h = sym.sin(kxy * x) * sym.cos(kxy * y) + sym.exp(-x - y)
    f_h = -(sym.diff(u_h, x, 2) + sym.diff(u_h, y, 2)) - k**2 * u_h
    registry["Helmholtz"] = PDESpec("Helmholtz", sym.lambdify((x, y), u_h, "numpy"), sym.lambdify((x, y), f_h, "numpy"))

    # 3) Variable coefficient
    a_sym = 2 + sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
    u_v = sym.sin(sym.pi * x) * sym.sin(sym.pi * y) + sym.exp(-x - y)
    f_v = -(sym.diff(a_sym * sym.diff(u_v, x), x) + sym.diff(a_sym * sym.diff(u_v, y), y))
    registry["VarCoeff"] = PDESpec("VarCoeff", sym.lambdify((x, y), u_v, "numpy"), sym.lambdify((x, y), f_v, "numpy"))

    # 4) HighFreq Poisson – sin(8πx)sin(8πy) + exp(-xy)
    u_hf = sym.sin(8 * sym.pi * x) * sym.sin(8 * sym.pi * y) + sym.exp(-x * y)
    f_hf = -(sym.diff(u_hf, x, 2) + sym.diff(u_hf, y, 2))
    registry["HighFreq"] = PDESpec("HighFreq", sym.lambdify((x, y), u_hf, "numpy"), sym.lambdify((x, y), f_hf, "numpy"))

    # 5) Sine-Gordon (Non-linear)
    u_sg = sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
    f_sg = sym.diff(u_sg, y, 2) - sym.diff(u_sg, x, 2) + sym.sin(u_sg)
    registry["SineGordon"] = PDESpec("SineGordon", sym.lambdify((x, y), u_sg, "numpy"), sym.lambdify((x, y), f_sg, "numpy"))

    # 6) KdV (Non-linear, 3rd-order derivative)
    u_kdv = sym.sin(sym.pi * x) * sym.cos(sym.pi * y)
    f_kdv = sym.diff(u_kdv, y, 1) + 6 * u_kdv * sym.diff(u_kdv, x, 1) + sym.diff(u_kdv, x, 3)
    registry["KdV"] = PDESpec("KdV", sym.lambdify((x, y), u_kdv, "numpy"), sym.lambdify((x, y), f_kdv, "numpy"))

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
# Random Feature Model Architecture
# ================================================================
class RandomFeatureModel(nn.Module):
    def __init__(self, d_in: int, N_features: int, sigma: float = 10.0):
        super().__init__()
        self.d_in = d_in
        self.N = N_features
        self.sigma = sigma
        
        # Frozen Random Features (W and b do NOT receive gradients)
        # W ~ N(0, sigma^2)
        self.W = nn.Parameter(torch.randn(d_in, N_features) * sigma, requires_grad=False)
        # b ~ U[-pi, pi]
        self.b = nn.Parameter(torch.rand(N_features) * 2 * np.pi - np.pi, requires_grad=False)
        
        # Trainable output coefficients
        self.c = nn.Parameter(torch.zeros(N_features, 1))
        nn.init.xavier_normal_(self.c)

    def forward(self, x):
        # Equation 4: phi(x) = (1 / sqrt(N)) * cos(<w, x> + b)
        phi = (1.0 / np.sqrt(self.N)) * torch.cos(torch.matmul(x, self.W) + self.b)
        return torch.matmul(phi, self.c)

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
# Solve one PDE using Random Feature Model
# ================================================================
def solve_rf_model(pde_name, width, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Note: For RF models, paper sets variance dynamically. 
    # sigma=10.0 (variance=100) is reported as optimal for their high-freq benchmarks.
    model = RandomFeatureModel(d_in=2, N_features=width, sigma=10.0).to(device)

    pts = torch.tensor(sample_interior(2000), device=device)
    bd  = torch.tensor(sample_boundary(300), device=device)

    u_bd_true = torch.tensor(
        PDE_REGISTRY[pde_name].u_fn(bd[:,0].cpu(), bd[:,1].cpu())
    ).reshape(-1,1).to(device)

    # We only pass `model.c` to the optimizer (W and b are frozen)
    optimizer_adam = torch.optim.Adam([model.c], lr=1e-2)
    
    # 1. Warm-up with Adam
    for _ in range(1000):
        optimizer_adam.zero_grad()
        r = pde_residual(model, pts, pde_name, PDE_REGISTRY[pde_name].f_fn)
        loss = (r**2).mean() + ((model(bd)-u_bd_true)**2).mean()
        loss.backward()
        optimizer_adam.step()

    # 2. Nonlinear Least Squares (RF-NLS) polishing via L-BFGS
    # Because there are only `width` parameters (e.g. 1024 max), L-BFGS is incredibly fast and optimal here.
    optimizer_lbfgs = torch.optim.LBFGS(
        [model.c],
        max_iter=1000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        r = pde_residual(model, pts, pde_name, PDE_REGISTRY[pde_name].f_fn)
        loss = (r**2).mean() + ((model(bd)-u_bd_true)**2).mean()
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    # Evaluation
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

print("\n--- Running Random Feature Model (RF-NLS) ---")
for pde in PDE_NAMES:
    for h in D_HIDDENS:
        for seed in SEEDS:
            mse = solve_rf_model(pde, h, seed)
            rmse = np.sqrt(mse)
            results[pde][h].append(rmse)
            print(f"[RF-NLS] {pde} N={h} seed={seed} RMSE={rmse:.2e}")

# ================================================================
# Statistics
# ================================================================
print("\n=== Random Feature Model (RF-NLS) Summary ===")
for pde in PDE_NAMES:
    print(f"\n[{pde}]")
    for h in D_HIDDENS:
        data = np.array(results[pde][h])
        res = bootstrap((data,), np.mean, confidence_level=0.95,
                        method="percentile", n_resamples=10_000, random_state=0)
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        ci_width = hi / 2 - lo / 2
        m = data.mean()
        print(f"  N={h:4d}  mean={m:.4e} ± {ci_width:.4e}")
        torch.cuda.empty_cache()