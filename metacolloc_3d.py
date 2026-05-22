"""
MetaColloc 3D: Zero-Shot Neural Basis Dictionary for 3D/Spatiotemporal PDEs

Key Innovations in this script:
  - Architecture: Dual-Branch SwiGLU (Low-freq raw + High-freq Fourier)
  - Dimensions: d_in = 3 (handles 3D spatial (x,y,z) or 2D+Time (x,y,t))
  - Test Suite: 
      1) Poisson3D (Smooth spatial)
      2) Burgers3D (Non-linear Convection-Diffusion, 2D+Time)
      3) AllenCahn3D (Non-linear Reaction-Diffusion, Phase Field, 2D+Time)
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
# PDE registry (3D / 2D+Time)
# ================================================================
@dataclass
class PDESpec:
    name: str
    u_fn: Callable
    f_fn: Callable


def _build_pde_registry() -> dict:
    x, y, z = sym.symbols("x y z") # 'z' acts as 't' for time-dependent PDEs
    registry = {}

    # 1) 3D Poisson: u_xx + u_yy + u_zz + f = 0
    u_p = sym.sin(sym.pi * x) * sym.sin(sym.pi * y) * sym.sin(sym.pi * z)
    f_p = -(sym.diff(u_p, x, 2) + sym.diff(u_p, y, 2) + sym.diff(u_p, z, 2))
    registry["Poisson3D"] = PDESpec(
        name="Poisson3D",
        u_fn=sym.lambdify((x, y, z), u_p, "numpy"),
        f_fn=sym.lambdify((x, y, z), f_p, "numpy"),
    )

    # 2) 2D+Time Burgers: u_z + u*u_x + u*u_y - nu*(u_xx + u_yy) = f
    nu_burgers = 0.01
    u_b = sym.sin(sym.pi * x) * sym.sin(sym.pi * y) * sym.exp(-z)
    f_b = sym.diff(u_b, z, 1) + u_b * sym.diff(u_b, x, 1) + u_b * sym.diff(u_b, y, 1) - nu_burgers * (sym.diff(u_b, x, 2) + sym.diff(u_b, y, 2))
    registry["Burgers3D"] = PDESpec(
        name="Burgers3D",
        u_fn=sym.lambdify((x, y, z), u_b, "numpy"),
        f_fn=sym.lambdify((x, y, z), f_b, "numpy"),
    )

    # 3) 2D+Time Allen-Cahn: u_z - nu*(u_xx + u_yy) - u*(1 - u^2) = f
    nu_ac = 0.001
    u_ac = sym.sin(sym.pi * x) * sym.sin(sym.pi * y) * sym.cos(sym.pi * z)
    f_ac = sym.diff(u_ac, z, 1) - nu_ac * (sym.diff(u_ac, x, 2) + sym.diff(u_ac, y, 2)) - u_ac * (1 - u_ac**2)
    registry["AllenCahn3D"] = PDESpec(
        name="AllenCahn3D",
        u_fn=sym.lambdify((x, y, z), u_ac, "numpy"),
        f_fn=sym.lambdify((x, y, z), f_ac, "numpy"),
    )

    return registry


PDE_REGISTRY = _build_pde_registry()
PDE_NAMES = list(PDE_REGISTRY.keys())
print(f"Registered 3D PDEs: {PDE_NAMES}")


# ================================================================
# Dual-Branch Model 
# ================================================================
class MetaColloc3D(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.freqs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] 
        
        # Low-freq branch
        d_low = d_hidden // 2
        self.fc1_low = nn.Linear(d_in, 4 * d_low)
        self.fc2_low = nn.Linear(2 * d_low, d_low)
        
        # High-freq branch
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


def build_model(d_hidden: int, d_in: int = 3) -> nn.Module:
    return MetaColloc3D(d_in, d_hidden).to(device)


def diff_lstsq(Phi: torch.Tensor, y: torch.Tensor, lam: float = 1e-4) -> torch.Tensor:
    H = Phi.shape[1]
    gram = Phi.T @ Phi + lam * torch.eye(H, device=Phi.device, dtype=Phi.dtype)
    return torch.linalg.solve(gram, Phi.T @ y)


# ================================================================
# GRF 3D task sampling
# ================================================================
def sample_multiscale_grf_3d(
    n_train: int, n_test: int, device: torch.device, D: int = 4096,
    rbf_prob: float = 0.40, hf_prob: float = 0.40, multi_prob: float = 0.20,
    ls_lo: float = 0.05, ls_hi: float = 0.5,
    hf_freq_lo: float = 5.0, hf_freq_hi: float = 50.0,
    hf_bw_lo: float = 1.0, hf_bw_hi: float = 10.0,
    dtype = torch.float32, d_in: int = 3
):
    M = n_train + n_test
    X = torch.rand((M, d_in), device=device, dtype=dtype)

    r = torch.rand(1).item()
    if r < rbf_prob: mode = "rbf"
    elif r < rbf_prob + hf_prob: mode = "highfreq"
    else: mode = "multi"

    def _rbf_omega(n_features: int) -> torch.Tensor:
        log_ls = torch.empty(1, device=device, dtype=dtype).uniform_(math.log(ls_lo), math.log(ls_hi))
        ls = torch.exp(log_ls).item()
        return torch.randn((n_features, d_in), device=device, dtype=dtype) / ls

    def _hf_omega(n_features: int) -> torch.Tensor:
        half = n_features // 2
        mu = torch.empty((1, d_in), device=device, dtype=dtype).uniform_(hf_freq_lo, hf_freq_hi)
        bw = torch.empty(1, device=device, dtype=dtype).uniform_(hf_bw_lo, hf_bw_hi).item()
        noise = torch.randn((half, d_in), device=device, dtype=dtype) * bw
        omega_pos = mu + noise
        omega_neg = -omega_pos
        return torch.cat([omega_pos, omega_neg], dim=0)

    if mode == "rbf": omega = _rbf_omega(D)
    elif mode == "highfreq": omega = _hf_omega(D)
    else: omega = torch.cat([_rbf_omega(D // 2), _hf_omega(D // 2)], dim=0)

    b = torch.empty((D,), device=device, dtype=dtype).uniform_(0, 2 * math.pi)
    w = torch.randn(D, device=device, dtype=dtype)

    projection = torch.matmul(X, omega.T) + b   
    phi = math.sqrt(2.0 / D) * torch.cos(projection)
    y   = torch.matmul(phi, w)                  

    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def sample_interior_3d(n: int) -> np.ndarray:
    return np.random.uniform(0, 1, (n, 3)).astype(np.float64)


def sample_boundary_3d(Q_per_face: int = 500) -> np.ndarray:
    # 3D has 6 boundary faces
    pts = []
    for d in range(3):
        for val in [0.0, 1.0]:
            face = np.random.uniform(0, 1, (Q_per_face, 3))
            face[:, d] = val
            pts.append(face)
    return np.vstack(pts)


# ================================================================
# Meta-training
# ================================================================
META_N_TRAIN     = 5000
META_N_TEST      = 2000
META_TASKS_EPOCH = 64
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

def meta_train(model, seed, checkpoint_dir="checkpoints_3d"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ext = ".safetensors" if _HAS_SAFETENSORS else ".pt"
    ckpt_path = os.path.join(checkpoint_dir, f"best_model_3d_h{model.d_hidden}_seed{seed}{ext}")

    model = model.to(dtype=TRAIN_DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=META_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=META_N_EPOCHS, eta_min=META_LR*0.01)

    best_val = float("inf")
    best_state = None

    model.train()
    for epoch in range(1, META_N_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        total_fit_loss = 0.0

        for _ in range(META_TASKS_EPOCH):
            X_tr, y_tr, X_te, y_te = sample_multiscale_grf_3d(META_N_TRAIN, META_N_TEST, device, dtype=TRAIN_DTYPE)
            Phi_tr, Phi_te = model(X_tr), model(X_te)
            w = diff_lstsq(Phi_tr, y_tr, META_LAM)
            total_fit_loss += F.mse_loss(Phi_te @ w, y_te)

        fit_loss_epoch = total_fit_loss / META_TASKS_EPOCH
        fit_loss_epoch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), META_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                X_tr, y_tr, X_te, y_te = sample_multiscale_grf_3d(META_N_TRAIN, META_N_TEST, device, dtype=TRAIN_DTYPE)
                w = diff_lstsq(model(X_tr), y_tr, META_LAM)
                val_loss = float(F.mse_loss(model(X_te) @ w, y_te).cpu())
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    _save_state_dict(ckpt_path, best_state)
            print(f"Epoch {epoch:03d} | fit={fit_loss_epoch.item():.3e} | val={val_loss:.3e}")

    model.load_state_dict(best_state if best_state is not None else _load_state_dict(ckpt_path))
    for p in model.parameters(): p.requires_grad_(False)
    return model.to(dtype=SOLVE_DTYPE).eval(), ckpt_path


# ================================================================
# Universal Newton-Raphson 3D PDE Solver 
# ================================================================
def precompute_bases_3d(model, pts, bd_pts):
    N, Nb = pts.shape[0], bd_pts.shape[0]
    H = model.d_hidden
    
    def get_phi(x): return model(x.unsqueeze(0)).squeeze(0)
    jac1 = jacfwd(get_phi)
    jac2 = jacfwd(jac1)

    Phi    = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_x  = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_y  = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_z  = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_xx = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_yy = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    Phi_zz = torch.empty((N, H), dtype=SOLVE_DTYPE, device=device)
    
    chunk = 64
    v_get_phi = vmap(get_phi)
    v_jac1, v_jac2 = vmap(jac1), vmap(jac2)

    pts_t = torch.tensor(pts, device=device, dtype=SOLVE_DTYPE)
    for i in range(0, N, chunk):
        b = min(chunk, N - i)
        x_b = pts_t[i : i + b]
        with torch.no_grad():
            Phi[i : i + b] = v_get_phi(x_b)
            
        j1 = v_jac1(x_b)
        Phi_x[i : i + b], Phi_y[i : i + b], Phi_z[i : i + b] = j1[..., 0], j1[..., 1], j1[..., 2]
        
        j2 = v_jac2(x_b)
        Phi_xx[i : i + b], Phi_yy[i : i + b], Phi_zz[i : i + b] = j2[..., 0, 0], j2[..., 1, 1], j2[..., 2, 2]

    bd_pts_t = torch.tensor(bd_pts, device=device, dtype=SOLVE_DTYPE)
    Phi_bd = torch.empty((Nb, H), dtype=SOLVE_DTYPE, device=device)
    for i in range(0, Nb, 256):
        b = min(256, Nb - i)
        with torch.no_grad():
            Phi_bd[i : i + b] = v_get_phi(bd_pts_t[i : i + b])

    return Phi, Phi_x, Phi_y, Phi_z, Phi_xx, Phi_yy, Phi_zz, Phi_bd


def solve_pde_3d(model, pde_spec: PDESpec, n_interior=8000, Q_boundary=600, n_test=20000):
    pts = sample_interior_3d(n_interior)
    bd_pts = sample_boundary_3d(Q_boundary) # 6 faces * 600 = 3600 points
    
    is_nonlinear = pde_spec.name in ["Burgers3D", "AllenCahn3D"]
    num_iters = 8 if is_nonlinear else 1
    
    Phi, Phi_x, Phi_y, Phi_z, Phi_xx, Phi_yy, Phi_zz, Phi_bd = precompute_bases_3d(model, pts, bd_pts)
    
    x_np, y_np, z_np = pts[:, 0], pts[:, 1], pts[:, 2]
    f_in = torch.tensor(pde_spec.f_fn(x_np, y_np, z_np).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
    
    bx, by, bz = bd_pts[:, 0], bd_pts[:, 1], bd_pts[:, 2]
    u_true_bd = torch.tensor(pde_spec.u_fn(bx, by, bz).reshape(-1, 1), dtype=SOLVE_DTYPE, device=device)
    
    w = torch.zeros((model.d_hidden, 1), dtype=SOLVE_DTYPE, device=device)

    for it in range(num_iters):
        u = Phi @ w
        u_xx, u_yy, u_zz = Phi_xx @ w, Phi_yy @ w, Phi_zz @ w
        u_x, u_y, u_z = Phi_x @ w, Phi_y @ w, Phi_z @ w
        
        if pde_spec.name == "Poisson3D":
            Res_in = -(u_xx + u_yy + u_zz) - f_in
            A_in = -(Phi_xx + Phi_yy + Phi_zz)
            
        elif pde_spec.name == "Burgers3D":
            nu = 0.01
            # L(u)[δu] = δu_z + u*δu_x + u_x*δu + u*δu_y + u_y*δu - nu*(Δδu)
            Res_in = u_z + u * u_x + u * u_y - nu * (u_xx + u_yy) - f_in
            A_in = Phi_z + u * Phi_x + u_x * Phi + u * Phi_y + u_y * Phi - nu * (Phi_xx + Phi_yy)
            
        elif pde_spec.name == "AllenCahn3D":
            nu = 0.001
            # L(u)[δu] = δu_z - nu*(Δδu) - (1 - 3u^2)*δu
            Res_in = u_z - nu * (u_xx + u_yy) - u * (1.0 - u**2) - f_in
            A_in = Phi_z - nu * (Phi_xx + Phi_yy) - (1.0 - 3.0 * u**2) * Phi

        Res_bd = (Phi_bd @ w) - u_true_bd
        A_bd = Phi_bd

        scale_in = 10.0 / (torch.amax(torch.abs(A_in), dim=1, keepdim=True) + 1e-12)
        scale_bd = 100.0 / (torch.amax(torch.abs(A_bd), dim=1, keepdim=True) + 1e-12)
        
        A_sys = torch.cat([A_in * scale_in, A_bd * scale_bd], dim=0)
        b_sys = torch.cat([-Res_in * scale_in, -Res_bd * scale_bd], dim=0)

        delta_w = torch.linalg.lstsq(A_sys, b_sys, rcond=None).solution
        w = w + delta_w

    xt = sample_interior_3d(n_test)
    with torch.no_grad():
        u_pred = (model(torch.tensor(xt, device=device, dtype=SOLVE_DTYPE)) @ w).cpu().numpy()
    u_true = pde_spec.u_fn(xt[:, 0], xt[:, 1], xt[:, 2]).reshape(-1, 1)
    return float(np.mean((u_pred - u_true) ** 2))

# ================================================================
# Run 3D Tests
# ================================================================
if __name__ == "__main__":
    SEEDS = list(range(42, 47))
    D_HIDDENS = [1024, 2048]

    results = {pde: {d: [] for d in D_HIDDENS} for pde in PDE_NAMES}
    times   = {pde: {d: [] for d in D_HIDDENS} for pde in PDE_NAMES}

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
            t_exp0 = time.time()
            model = build_model(d_hidden)
            model, _ = meta_train(model, seed=seed)

            for pde_name in PDE_NAMES:
                mse = solve_pde_3d(model, PDE_REGISTRY[pde_name])
                rmse = np.sqrt(mse)
                results[pde_name][d_hidden].append(rmse)
            t_exp = time.time() - t_exp0
            for pde_name in PDE_NAMES:
                times[pde_name][d_hidden].append(t_exp / len(PDE_NAMES))
            elapsed = time.time() - t0
            pbar.set_postfix({
                "H": d_hidden, "seed": seed,
                "Poisson": f"{results['Poisson3D'][d_hidden][-1]:.2e}",
                "elapsed": f"{elapsed / 60:.1f}min",
            })
            pbar.update(1)
    pbar.close()
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    stats = {pde: {} for pde in PDE_NAMES}
    print("\n=== Summary ===")
    for pde_name in PDE_NAMES:
        print(f"  [{pde_name}]")
        for d_hidden in D_HIDDENS:
            data = np.asarray(results[pde_name][d_hidden], dtype=np.float64)
            m = data.mean()
            res = bootstrap((data,), np.mean, confidence_level=0.95, method="percentile", n_resamples=10_000, random_state=0)
            lo, hi = res.confidence_interval.low, res.confidence_interval.high
            ci_width = (hi - lo) / 2
            stats[pde_name][d_hidden] = {"mean": m, "lo": lo, "hi": hi}
            mean_t = np.mean(times[pde_name][d_hidden])
            print(f"    H={d_hidden:4d}  mean={m:.4e} ± {ci_width:.4e}")