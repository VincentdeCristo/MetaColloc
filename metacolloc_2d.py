"""
MetaColloc

  - Architecture: SwiGLU FFN
  - Solver: Universal Newton-Raphson closed-form lstsq solver.
  - Test Suite: 
      1) Poisson (Smooth)
      2) Helmholtz (High-Freq, k=64π)
      3) VarCoeff (Variable Coefficients)
      4) HighFreq (sin(8πx)sin(8πy))
      5) Sine-Gordon (Non-linear, 1D+1D)
      6) KdV (Non-linear, 3rd-order derivative, 1D+1D)
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import bootstrap
from torch.func import vmap, jacfwd
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_DTYPE = torch.float64
SOLVE_DTYPE = torch.float64
torch.set_default_dtype(TRAIN_DTYPE)


try:
    from safetensors.torch import save_file as safetensors_save
    from safetensors.torch import load_file as safetensors_load
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False
    safetensors_save = None
    safetensors_load = None


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
# Model (Enhanced with Multi-scale Fourier Features)
# ================================================================
class MetaColloc(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.freqs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] 
        
        d_low = d_hidden // 2
        self.fc1_low = nn.Linear(d_in, 4 * d_low)
        self.fc2_low = nn.Linear(2 * d_low, d_low)
        
        d_high_in = d_in * 2 * len(self.freqs)
        d_high = d_hidden - d_low
        self.fc1_high = nn.Linear(d_high_in, 4 * d_high)
        self.fc2_high = nn.Linear(2 * d_high, d_high)

    def forward(self, x):
        h_low = self.fc1_low(x)
        h1_low, h2_low = h_low.chunk(2, dim=-1)
        h_low = self.fc2_low(h1_low * F.silu(h2_low))
        
        features = []
        for k in self.freqs:
            freq = k * torch.pi
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        x_high = torch.cat(features, dim=-1)
        
        h_high = self.fc1_high(x_high)
        h1_high, h2_high = h_high.chunk(2, dim=-1)
        h_high = self.fc2_high(h1_high * F.silu(h2_high))
        
        return torch.cat([h_low, h_high], dim=-1)


def build_model(d_hidden: int, d_in: int = 2) -> nn.Module:
    return MetaColloc(d_in, d_hidden).to(device)


def diff_lstsq(Phi: torch.Tensor, y: torch.Tensor, lam: float = 1e-4) -> torch.Tensor:
    H = Phi.shape[1]
    gram = Phi.T @ Phi + lam * torch.eye(H, device=Phi.device, dtype=Phi.dtype)
    return torch.linalg.solve(gram, Phi.T @ y)


# ================================================================
# GRF task sampling
# ================================================================
def sample_multiscale_grf(
    n_train: int,
    n_test:  int,
    device:  torch.device,
    D:       int   = 4096,
    rbf_prob:   float = 0.40,
    hf_prob:    float = 0.40,
    multi_prob: float = 0.20,
    ls_lo: float = 0.005,  
    ls_hi: float = 0.05,
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
        omega_pos = mu + noise          
        omega_neg = -omega_pos          
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

def sample_interior(n: int) -> np.ndarray:
    return np.random.uniform(0, 1, (n, 2)).astype(np.float64)


def sample_boundary(Q: int = 100) -> np.ndarray:
    t = np.linspace(0, 1, Q, dtype=np.float64)
    return np.vstack([
        np.stack([t, np.zeros_like(t)], 1),
        np.stack([t, np.ones_like(t)], 1),
        np.stack([np.zeros_like(t), t], 1),
        np.stack([np.ones_like(t), t], 1)
    ])


# ================================================================
# Meta-training configuration
# ================================================================
META_N_TRAIN     = 4000
META_N_TEST      = 1500
META_TASKS_EPOCH = 128
META_N_EPOCHS    = 1000
META_LR          = 1e-3
META_LAM         = 1e-8
META_GRAD_CLIP   = 1.0


def _save_state_dict(path: str, state_dict: dict):
    if _HAS_SAFETENSORS: safetensors_save(state_dict, path)
    else: torch.save(state_dict, path)


def _load_state_dict(path: str):
    if path.endswith(".safetensors") and _HAS_SAFETENSORS: return safetensors_load(path)
    return torch.load(path, map_location="cpu")


def meta_train(
    model: nn.Module, seed: int, n_epochs=META_N_EPOCHS,
    tasks_epoch=META_TASKS_EPOCH, checkpoint_dir="checkpoints"
) -> tuple[nn.Module, str]:
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    ext = ".safetensors" if _HAS_SAFETENSORS else ".pt"
    ckpt_path = os.path.join(checkpoint_dir, f"best_model_d{model.d_in}_h{model.d_hidden}_seed{seed}{ext}")

    model = model.to(dtype=TRAIN_DTYPE)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=META_LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=20)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 20, eta_min=META_LR * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[20])

    best_val = float("inf")
    best_state = None

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        total_fit_loss = torch.zeros((), device=device, dtype=TRAIN_DTYPE)

        for _ in range(tasks_epoch):
            X_tr_t, y_tr_t, X_te_t, y_te_t = sample_multiscale_grf(META_N_TRAIN, META_N_TEST, device, dtype=TRAIN_DTYPE)

            Phi_tr, Phi_te = model(X_tr_t), model(X_te_t)

            w = diff_lstsq(Phi_tr, y_tr_t, META_LAM)
            total_fit_loss = total_fit_loss + F.mse_loss(Phi_te @ w, y_te_t)

        fit_loss_epoch = total_fit_loss / tasks_epoch
        fit_loss_epoch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), META_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(8):
                X_tr_t, y_tr_t, X_te_t, y_te_t = sample_multiscale_grf(META_N_TRAIN, META_N_TEST, device, dtype=TRAIN_DTYPE)

                Phi_tr, Phi_te = model(X_tr_t), model(X_te_t)

                w = diff_lstsq(Phi_tr, y_tr_t, META_LAM)
                val_loss += float(F.mse_loss(Phi_te @ w, y_te_t).cpu())

        val_loss /= 8.0

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_state_dict(ckpt_path, best_state)

        model.train()
        # if epoch == 1 or epoch % max(1, n_epochs // 10) == 0:
        #     print(f"Epoch {epoch:03d}/{n_epochs} | fit={fit_loss_epoch.item():.3e} | val={val_loss:.3e} | lr={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state if best_state is not None else _load_state_dict(ckpt_path))
    for p in model.parameters(): p.requires_grad_(False)
    model = model.to(dtype=SOLVE_DTYPE).eval()
    return model, ckpt_path


# ================================================================
# Universal Newton-Raphson PDE Solver (Closed-form lstsq iterations)
# ================================================================
def precompute_bases(model, pts, bd_pts, needs_third=False):
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
    num_iters = 64 if is_nonlinear else 1
    
    Phi, Phi_x, Phi_y, Phi_xx, Phi_yy, Phi_xxx, Phi_bd = precompute_bases(model, pts, bd_pts, needs_third)
    
    pts_x_np, pts_y_np = pts[:, 0], pts[:, 1]
    f_in = torch.tensor(pde_spec.f_fn(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
    
    bd_x_np, bd_y_np = bd_pts[:, 0], bd_pts[:, 1]
    u_true_bd = torch.tensor(pde_spec.u_fn(bd_x_np, bd_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
    
    H = model.d_hidden
    w = torch.zeros((H, 1), dtype=SOLVE_DTYPE, device=device)
    
    if pde_spec.name == "VarCoeff":
        a_t = torch.tensor(_a_varcoeff(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
        ax_t = torch.tensor(_ax_varcoeff(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
        ay_t = torch.tensor(_ay_varcoeff(pts_x_np, pts_y_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)

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

        scale_in = 100.0 / (torch.amax(torch.abs(A_in), dim=1, keepdim=True) + 1e-12)
        scale_bd = 100.0 / (torch.amax(torch.abs(A_bd), dim=1, keepdim=True) + 1e-12)
        
        A_sys = torch.cat([A_in * scale_in, A_bd * scale_bd], dim=0)
        b_sys = torch.cat([-Res_in * scale_in, -Res_bd * scale_bd], dim=0)

        delta_w = torch.linalg.lstsq(A_sys, b_sys, rcond=None).solution
        w = w + delta_w

    xt = sample_interior(n_test)
    with torch.no_grad():
        u_pred = (model(torch.tensor(xt, device=device, dtype=SOLVE_DTYPE)) @ w).cpu().numpy()
    u_true = pde_spec.u_fn(xt[:, 0], xt[:, 1]).reshape(-1, 1)
    return float(np.mean((u_pred - u_true) ** 2))


# ================================================================
# Helper: Compute Mean and 95% CI
# ================================================================
def compute_stats(data_list):
    data = np.asarray(data_list, dtype=np.float64)
    m = data.mean()
    if len(data) < 2 or np.all(data == data[0]):
        return m, m, m
    try:
        res = bootstrap((data,), np.mean, confidence_level=0.95, method="percentile", n_resamples=10_000, random_state=0)
        return m, res.confidence_interval.low, res.confidence_interval.high
    except Exception:
        return m, m, m


# ================================================================
# Main Experiment Loop
# ================================================================
if __name__ == "__main__":
    SEEDS        = list(range(42, 47))
    D_HIDDENS    = [128, 256, 512, 1024]

    results      = {pde: {d: [] for d in D_HIDDENS} for pde in PDE_NAMES}
    train_times  = {d: [] for d in D_HIDDENS}
    solve_times  = {pde: {d: [] for d in D_HIDDENS} for pde in PDE_NAMES}

    total = len(D_HIDDENS) * len(SEEDS)
    pbar  = tqdm(total=total, desc="Experiments", ncols=120)
    t0    = time.time()

    for d_hidden in D_HIDDENS:
        for seed in SEEDS:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            t_train_start = time.time()
            model = build_model(d_hidden)
            model, ckpt_path = meta_train(
                model, seed,
                checkpoint_dir=os.path.join("checkpoints")
            )
            train_times[d_hidden].append(time.time() - t_train_start)

            for pde_name in PDE_NAMES:
                t_solve_start = time.time()
                mse = solve_pde(model, PDE_REGISTRY[pde_name])
                solve_times[pde_name][d_hidden].append(time.time() - t_solve_start)
                
                rmse = np.sqrt(mse)
                results[pde_name][d_hidden].append(rmse)

            elapsed = time.time() - t0
            pbar.set_postfix({
                "H": d_hidden, "seed": seed,
                "Poisson": f"{results['Poisson'][d_hidden][-1]:.2e}",
                "elapsed": f"{elapsed / 60:.1f}min",
            })
            pbar.update(1)

    pbar.close()
    print(f"Total Experiment Time: {(time.time() - t0) / 60:.1f} min")

    # ================================================================
    # Reporting Sections
    # ================================================================
    
    print("\n" + "="*60)
    print("=== 1. Performance Summary (RMSE: Mean ± 95% CI) ===")
    print("="*60)
    for pde_name in PDE_NAMES:
        print(f"  [{pde_name}]")
        for d_hidden in D_HIDDENS:
            m, lo, hi = compute_stats(results[pde_name][d_hidden])
            ci_width = (hi - lo) / 2
            print(f"    H={d_hidden:4d}  RMSE = {m:.4e} ± {ci_width:.4e}")

    print("\n" + "="*60)
    print("=== 2. Training Time Summary (Seconds: Mean ± 95% CI) ===")
    print("="*60)
    for d_hidden in D_HIDDENS:
        m, lo, hi = compute_stats(train_times[d_hidden])
        ci_width = (hi - lo) / 2
        print(f"  H={d_hidden:4d}  Time = {m:.2f}s ± {ci_width:.2f}s")

    print("\n" + "="*60)
    print("=== 3. Solving Time Summary (Seconds: Mean ± 95% CI) ===")
    print("="*60)
    for pde_name in PDE_NAMES:
        print(f"  [{pde_name}]")
        for d_hidden in D_HIDDENS:
            m, lo, hi = compute_stats(solve_times[pde_name][d_hidden])
            ci_width = (hi - lo) / 2
            print(f"    H={d_hidden:4d}  Time = {m:.3f}s  ± {ci_width:.3f}s")