"""
MetaColloc's "Non-Learning" Control Group (Random Features + γ Tuning)
"""

import os, warnings, time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import lstsq
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
import optuna
from tqdm import tqdm
from scipy.stats import bootstrap
import sympy as sym
from dataclasses import dataclass
from typing import Callable
from torch.func import vmap, jacfwd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.set_default_dtype(torch.float64)
SOLVE_DTYPE = torch.float64

# ================================================================
# PDE registry
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
# Model
# ================================================================
class MetaColloc(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, gamma: float):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.freqs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] 
        
        d_low = d_hidden // 2
        self.fc1_low = nn.Linear(d_in, 4 * d_low)
        self.fc2_low = nn.Linear(2 * d_low, d_low)
        nn.init.normal_(self.fc1_low.weight)
        nn.init.normal_(self.fc2_low.weight)
        self.fc1_low.weight.data = F.normalize(self.fc1_low.weight.data, p=2, dim=1)
        self.fc2_low.weight.data = F.normalize(self.fc2_low.weight.data, p=2, dim=1)
        nn.init.uniform_(self.fc1_low.bias, 0, 1)
        nn.init.uniform_(self.fc2_low.bias, 0, 1)
        self.gamma = gamma
        
        d_high_in = d_in * 2 * len(self.freqs)
        d_high = d_hidden - d_low
        self.fc1_high = nn.Linear(d_high_in, 4 * d_high)
        self.fc2_high = nn.Linear(2 * d_high, d_high)
        nn.init.normal_(self.fc1_high.weight)
        nn.init.normal_(self.fc2_high.weight)
        self.fc1_high.weight.data = F.normalize(self.fc1_high.weight.data, p=2, dim=1)
        self.fc2_high.weight.data = F.normalize(self.fc2_high.weight.data, p=2, dim=1)
        nn.init.uniform_(self.fc1_high.bias, 0, 1)
        nn.init.uniform_(self.fc2_high.bias, 0, 1)
    
    def forward(self, x):
        h_low = self.gamma * self.fc1_low(x)
        h1_low, h2_low = h_low.chunk(2, dim=-1)
        h_low = self.gamma * self.fc2_low(h1_low * F.silu(h2_low))
        
        features = []
        for k in self.freqs:
            freq = k * torch.pi
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        x_high = torch.cat(features, dim=-1)
        
        h_high = self.gamma * self.fc1_high(x_high)
        h1_high, h2_high = h_high.chunk(2, dim=-1)
        h_high = self.gamma * self.fc2_high(h1_high * F.silu(h2_high))
        
        return torch.cat([h_low, h_high], dim=-1)


def build_model(d_hidden: int, gamma: float, d_in: int = 2) -> nn.Module:
    return MetaColloc(d_in, d_hidden, gamma).to(device)

# ================================================================
# Sampling, GP Proxy Tasks, & γ-Tuning
# ================================================================
def sample_interior(n: int) -> np.ndarray:
    return np.random.uniform(0, 1, (n, 2)).astype(np.float64)

def sample_boundary(Q: int = 100) -> np.ndarray:
    t = np.linspace(0, 1, Q, dtype=np.float64)
    bottom = np.stack([t, np.zeros_like(t)], 1)
    top    = np.stack([t, np.ones_like(t)], 1)
    left   = np.stack([np.zeros_like(t), t], 1)
    right  = np.stack([np.ones_like(t), t], 1)
    return np.vstack([bottom, top, left, right])

def sample_multiscale_grf(
    n_train: int,
    n_test:  int,
    device:  torch.device,
    D:       int   = 4096,
    rbf_prob:   float = 0.40,
    hf_prob:    float = 0.40,
    multi_prob: float = 0.20,
    ls_lo: float = 0.01,  
    ls_hi: float = 0.10,
    hf_freq_lo:    float = 10.0,
    hf_freq_hi:    float = 300.0,
    hf_bw_lo:      float = 1.0,
    hf_bw_hi:      float = 15.0,
    dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    M = n_train + n_test
    X = torch.rand((M, 2), device=device, dtype=dtype)

    r = torch.rand(1).item()
    if r < rbf_prob:
        mode = "rbf"
    elif r < rbf_prob + hf_prob:
        mode = "highfreq"
    else:
        mode = "multi"

    def _rbf_omega(n_features: int) -> torch.Tensor:
        log_ls = torch.empty(1, device=device, dtype=dtype).uniform_(
            math.log(ls_lo), math.log(ls_hi)
        )
        ls = torch.exp(log_ls).item()
        return torch.randn((n_features, 2), device=device, dtype=dtype) / ls

    def _hf_omega(n_features: int) -> torch.Tensor:
        half = n_features // 2
        mu = torch.empty((1, 2), device=device, dtype=dtype).uniform_(
            hf_freq_lo, hf_freq_hi
        )
        bw = torch.empty(1, device=device, dtype=dtype).uniform_(
            hf_bw_lo, hf_bw_hi
        ).item()
        
        noise = torch.randn((half, 2), device=device, dtype=dtype) * bw
        omega_pos = mu + noise          # N(+μ, σ²)
        omega_neg = -omega_pos          # N(-μ, σ²)
        return torch.cat([omega_pos, omega_neg], dim=0)

    if mode == "rbf":
        omega = _rbf_omega(D)
    elif mode == "highfreq":
        omega = _hf_omega(D)
    else:  
        omega = torch.cat([_rbf_omega(D // 2), _hf_omega(D // 2)], dim=0)

    b = torch.empty((D,), device=device, dtype=dtype).uniform_(0, 2 * math.pi)
    w = torch.randn(D, device=device, dtype=dtype)

    projection = torch.matmul(X, omega.T) + b   
    phi = math.sqrt(2.0 / D) * torch.cos(projection)
    y   = torch.matmul(phi, w)                  

    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

def gp_fitting_mse(
    d_hidden: int, gamma: float, seed: int,
    n_train: int = 4000, n_test: int = 1500
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_tr, y_tr, X_te, y_te = sample_multiscale_grf(
        n_train=n_train, n_test=n_test, device=device, dtype=SOLVE_DTYPE
    )

    try:
        model = build_model(d_hidden, gamma)
        model.eval()
        with torch.no_grad():
            phi_tr = model(X_tr)
            w = lstsq(phi_tr, y_tr, rcond=None).solution
            phi_te = model(X_te)
            y_pred = (phi_te @ w).cpu().numpy()
        mse = float(np.mean((y_pred - y_te.cpu().numpy()) ** 2))
        return mse if np.isfinite(mse) else 1e6
    except Exception:
        return 1e6

def tune_gamma(d_hidden: int, seed: int, n_trials: int = 100,
               gamma_lo: float = 0.1, gamma_hi: float = 10.0) -> float:
    def objective(trial):
        gamma = trial.suggest_float("gamma", gamma_lo, gamma_hi, log=True)
        return gp_fitting_mse(d_hidden, gamma, seed)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return float(study.best_params["gamma"])

# ================================================================
# Universal Newton-Raphson PDE Solver (Closed-form lstsq iterations)
# ================================================================
def precompute_bases(model, pts, bd_pts, needs_third=False):
    """Memory-efficient precomputation of basis matrix and its derivatives using jacfwd"""
    N, Nb = pts.shape[0], bd_pts.shape[0]
    H = model.d_hidden
    
    def get_phi(x): return model(x.unsqueeze(0)).squeeze(0)
    jac1 = jacfwd(get_phi)
    jac2 = jacfwd(jac1)
    jac3 = jacfwd(jac2) if needs_third else None

    Phi     = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_x   = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_y   = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_xx  = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_yy  = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_xxx = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device) if needs_third else None
    
    # 3rd-order jacfwd can be memory intensive, use smaller chunk if needed
    chunk = 16 if needs_third else 64
    v_get_phi = vmap(get_phi)
    v_jac1, v_jac2 = vmap(jac1), vmap(jac2)
    v_jac3 = vmap(jac3) if needs_third else None

    pts_t = torch.tensor(pts, device=device, dtype=SOLVE_DTYPE)
    for i in range(0, N, chunk):
        b = min(chunk, N - i)
        x_b = pts_t[i : i + b]
        
        with torch.no_grad():
            Phi[i : i + b] = v_get_phi(x_b)
            
        j1 = v_jac1(x_b)
        Phi_x[i : i + b] = j1[..., 0]
        Phi_y[i : i + b] = j1[..., 1]
        
        j2 = v_jac2(x_b)
        Phi_xx[i : i + b] = j2[..., 0, 0]
        Phi_yy[i : i + b] = j2[..., 1, 1]
        
        if needs_third:
            Phi_xxx[i : i + b] = v_jac3(x_b)[..., 0, 0, 0]

    bd_pts_t = torch.tensor(bd_pts, device=device, dtype=SOLVE_DTYPE)
    Phi_bd = torch.empty((Nb, H), dtype=SOLVE_DTYPE, device=device)
    for i in range(0, Nb, 256):
        b = min(256, Nb - i)
        with torch.no_grad():
            Phi_bd[i : i + b] = v_get_phi(bd_pts_t[i : i + b])

    return Phi, Phi_x, Phi_y, Phi_xx, Phi_yy, Phi_xxx, Phi_bd


def solve_pde(model, pde_spec: PDESpec, n_interior=2000, Q_boundary=300, n_test=10000):
    pts = sample_interior(n_interior)
    bd_pts = sample_boundary(Q_boundary)
    
    needs_third = (pde_spec.name == "KdV")
    is_nonlinear = pde_spec.name in ["SineGordon", "KdV"]
    num_iters = 32 if is_nonlinear else 1
    
    Phi, Phi_x, Phi_y, Phi_xx, Phi_yy, Phi_xxx, Phi_bd = precompute_bases(model, pts, bd_pts, needs_third)
    
    # Precompute target f and boundary True values
    pts_x_np, pts_y_np = pts[:, 0], pts[:, 1]
    f_in = torch.tensor(pde_spec.f_fn(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
    
    bd_x_np, bd_y_np = bd_pts[:, 0], bd_pts[:, 1]
    u_true_bd = torch.tensor(pde_spec.u_fn(bd_x_np, bd_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
    
    H = model.d_hidden
    w = torch.zeros((H, 1), dtype=SOLVE_DTYPE, device=device)
    
    # Precompute VarCoeff arrays to avoid doing it inside the loop
    if pde_spec.name == "VarCoeff":
        a_t = torch.tensor(_a_varcoeff(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
        ax_t = torch.tensor(_ax_varcoeff(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
        ay_t = torch.tensor(_ay_varcoeff(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)

    # Newton-Raphson Iteration Loop (Solves A * delta_w = -Res)
    for it in range(num_iters):
        u = Phi @ w
        u_xx, u_yy = Phi_xx @ w, Phi_yy @ w
        
        if pde_spec.name in ["Poisson", "HighFreq"]:
            Res_in = -(u_xx + u_yy) - f_in
            A_in = -(Phi_xx + Phi_yy)
            
        elif pde_spec.name == "Helmholtz":
            Res_in = -(u_xx + u_yy) - _HELMHOLTZ_K2 * u - f_in
            A_in = -(Phi_xx + Phi_yy) - _HELMHOLTZ_K2 * Phi
            
        elif pde_spec.name == "VarCoeff":
            u_x, u_y = Phi_x @ w, Phi_y @ w
            Res_in = -(a_t * (u_xx + u_yy) + ax_t * u_x + ay_t * u_y) - f_in
            A_in = -(a_t * (Phi_xx + Phi_yy) + ax_t * Phi_x + ay_t * Phi_y)
            
        elif pde_spec.name == "SineGordon":
            Res_in = u_yy - u_xx + torch.sin(u) - f_in
            A_in = Phi_yy - Phi_xx + torch.cos(u) * Phi
            
        elif pde_spec.name == "KdV":
            u_x, u_y = Phi_x @ w, Phi_y @ w
            u_xxx = Phi_xxx @ w
            Res_in = u_y + 6 * u * u_x + u_xxx - f_in
            A_in = Phi_y + 6 * u * Phi_x + 6 * u_x * Phi + Phi_xxx

        Res_bd = (Phi_bd @ w) - u_true_bd
        A_bd = Phi_bd

        # Row scaling to prevent condition number explosion
        scale_in = 100.0 / (torch.amax(torch.abs(A_in), dim=1, keepdim=True) + 1e-12)
        scale_bd = 100.0 / (torch.amax(torch.abs(A_bd), dim=1, keepdim=True) + 1e-12)
        
        A_sys = torch.cat([A_in * scale_in, A_bd * scale_bd], dim=0)
        b_sys = torch.cat([-Res_in * scale_in, -Res_bd * scale_bd], dim=0)

        delta_w = torch.linalg.lstsq(A_sys, b_sys, rcond=None).solution
        w = w + delta_w

    # Evaluate on test set
    xt = sample_interior(n_test)
    with torch.no_grad():
        u_pred = (model(torch.tensor(xt, device=device, dtype=SOLVE_DTYPE)) @ w).cpu().numpy()
    u_true = pde_spec.u_fn(xt[:, 0], xt[:, 1]).reshape(-1, 1)
    return float(np.mean((u_pred - u_true) ** 2))

# ================================================================
# main experiment loop
# ================================================================
SEEDS = list(range(42, 47))
D_HIDDENS = [128, 256, 512, 1024]
N_TRIALS = 100

results = {p: {d: [] for d in D_HIDDENS} for p in PDE_NAMES}
best_gammas = {}
times = {p: {d: [] for d in D_HIDDENS} for p in PDE_NAMES}

total = len(D_HIDDENS) * len(SEEDS) * len(PDE_NAMES)
pbar = tqdm(total=total, desc="Experiments", ncols=140)
t0 = time.time()

for d_hidden in D_HIDDENS:
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        gamma = tune_gamma(d_hidden, seed, n_trials=N_TRIALS)
        best_gammas[(d_hidden, seed)] = gamma

        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_model(d_hidden, gamma)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        for pde_name in PDE_NAMES:
            np.random.seed(seed)
            torch.manual_seed(seed)
            t_solve0 = time.time()
            mse = solve_pde(model, PDE_REGISTRY[pde_name])
            rmse = np.sqrt(mse)
            t_solve = time.time() - t_solve0

            results[pde_name][d_hidden].append(rmse)
            times[pde_name][d_hidden].append(t_solve)

            elapsed = time.time() - t0
            pbar.set_postfix({
                "H": d_hidden, "seed": seed,
                "pde": pde_name[:8], "γ*": f"{gamma:.2f}",
                "rmse": f"{rmse:.2e}", "elapsed": f"{elapsed/60:.1f}min"
            })
            pbar.update(1)

pbar.close()
print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

# ================================================================
# BCa CI
# ================================================================
stats = {p: {} for p in PDE_NAMES}
print("\n=== Summary ===")
for pde_name in PDE_NAMES:
    print(f"  [{pde_name}]")
    for d_hidden in D_HIDDENS:
        data = np.array(results[pde_name][d_hidden])
        m = data.mean()
        res = bootstrap((data,), np.mean, confidence_level=0.95,
                        method="percentile", n_resamples=10_000, random_state=0)
        lo = res.confidence_interval.low
        hi = res.confidence_interval.high
        ci_width = (hi - lo) / 2
        stats[pde_name][d_hidden] = {"mean": m, "lo": lo, "hi": hi}
        mean_t = np.mean(times[pde_name][d_hidden])
        print(f"    H={d_hidden:4d}  mean={m:.4e} ± {ci_width:.4e}")