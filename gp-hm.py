"""
GP-HM-StM: Gaussian Process solver with Student-t Spectral Mixture kernel
for 6 custom PDEs (Poisson, Helmholtz, VarCoeff, HighFreq, SineGordon, KdV).

Changes vs GP-HM-GM baseline:
  1. SpectralMixture1D (Gaussian envelope) → StMixture1D (Matérn-5/2 envelope)
  2. F selected per-PDE from {20, 40, 100} via quick probe (seed=42, 15k steps)

Analytic derivatives of k(z) = Σ_q w_q · γ(|z|; ρ_q) · cos(2π μ_q z)
where γ is Matérn-5/2 with α = √5/ρ:
  dγ/dz   = −(α²/3) z (1 + αr) E
  d²γ/dz² = −(α²/3)(1 + αr − α²r²) E
  d³γ/dz³ =  (α⁴/3) z (3 − αr) E
Full kernel chain rule (m = 2πμ):
  dk/dz   = dγ·C − m·γ·S
  d²k/dz² = d²γ·C − 2m·dγ·S − m²·γ·C
  d³k/dz³ = d³γ·C − 3m·d²γ·S − 3m²·dγ·C + m³·γ·S
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import sympy as sym
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap
import time

torch.set_default_dtype(torch.float64)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# 1. PDE definitions
# ================================================================
@dataclass
class PDESpec:
    name: str
    u_fn: Callable
    f_fn: Callable

def build_pde_registry() -> Dict[str, PDESpec]:
    x, y = sym.symbols("x y")
    reg = {}

    u = sym.sin(2*sym.pi*x)*sym.sin(2*sym.pi*y) + sym.exp(-x-y)
    reg["Poisson"] = PDESpec("Poisson",
        sym.lambdify((x,y), u, "numpy"),
        sym.lambdify((x,y), -(sym.diff(u,x,2)+sym.diff(u,y,2)), "numpy"))

    k_helm = 64*sym.pi
    u_h = sym.sin(k_helm/sym.sqrt(2)*x)*sym.cos(k_helm/sym.sqrt(2)*y)+sym.exp(-x-y)
    reg["Helmholtz"] = PDESpec("Helmholtz",
        sym.lambdify((x,y), u_h, "numpy"),
        sym.lambdify((x,y), -(sym.diff(u_h,x,2)+sym.diff(u_h,y,2))-k_helm**2*u_h, "numpy"))

    a_s = 2+sym.sin(sym.pi*x)*sym.cos(sym.pi*y)
    u_v = sym.sin(sym.pi*x)*sym.sin(sym.pi*y)+sym.exp(-x-y)
    reg["VarCoeff"] = PDESpec("VarCoeff",
        sym.lambdify((x,y), u_v, "numpy"),
        sym.lambdify((x,y), -(sym.diff(a_s*sym.diff(u_v,x),x)+sym.diff(a_s*sym.diff(u_v,y),y)), "numpy"))

    u_hf = sym.sin(8*sym.pi*x)*sym.sin(8*sym.pi*y)+sym.exp(-x*y)
    reg["HighFreq"] = PDESpec("HighFreq",
        sym.lambdify((x,y), u_hf, "numpy"),
        sym.lambdify((x,y), -(sym.diff(u_hf,x,2)+sym.diff(u_hf,y,2)), "numpy"))

    u_sg = sym.sin(sym.pi*x)*sym.cos(sym.pi*y)
    reg["SineGordon"] = PDESpec("SineGordon",
        sym.lambdify((x,y), u_sg, "numpy"),
        sym.lambdify((x,y), sym.diff(u_sg,y,2)-sym.diff(u_sg,x,2)+sym.sin(u_sg), "numpy"))

    u_k = sym.sin(sym.pi*x)*sym.cos(sym.pi*y)
    reg["KdV"] = PDESpec("KdV",
        sym.lambdify((x,y), u_k, "numpy"),
        sym.lambdify((x,y), sym.diff(u_k,y)+6*u_k*sym.diff(u_k,x)+sym.diff(u_k,x,3), "numpy"))

    return reg

PDE_REGISTRY = build_pde_registry()
PDE_NAMES    = list(PDE_REGISTRY.keys())

_HELMHOLTZ_K2 = (64*np.pi)**2
_a_vc  = lambda x,y: 2 + np.sin(np.pi*x)*np.cos(np.pi*y)
_ax_vc = lambda x,y: np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)
_ay_vc = lambda x,y: -np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)

def sample_interior(n: int) -> np.ndarray:
    return np.random.rand(n, 2)

# ================================================================
# 2. StM Kernel (Matérn-5/2 spectral mixture) + analytic derivatives
# ================================================================
def safe_cholesky(A: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    return torch.linalg.cholesky(
        A + jitter * torch.eye(A.shape[0], dtype=A.dtype, device=A.device))

class StMixture1D(nn.Module):
    """
    k(z) = Σ_q w_q · Matérn52(|z|; ρ_q) · cos(2π μ_q z)

    All four tensor-valued kernels (K, dK, d²K, d³K) are computed in closed form.
    Diagonal (z=0) is handled naturally: dγ/dz=0, d³γ/dz³=0 at z=0 (z factor).
    """
    def __init__(self, Q: int = 30, F: float = 100.0):
        super().__init__()
        self.Q = Q; self.F = F
        self.log_w   = nn.Parameter(torch.full((Q,), -np.log(Q)))
        self.mu      = nn.Parameter(torch.linspace(0, F, Q))
        self.log_rho = nn.Parameter(torch.zeros(Q))   # log length-scale

    def get_K_dK_d2K_d3K(self, h: torch.Tensor, order: int = 3):
        """
        h : (M,) grid points
        order : highest derivative needed (0/1/2/3)
        Returns (K, dK, d2K, d3K), unused slots are None.
        """
        w   = torch.exp(self.log_w).view(1,1,-1)   # (1,1,Q)
        rho = torch.exp(self.log_rho).view(1,1,-1)  # (1,1,Q)
        mu  = self.mu.view(1,1,-1)                  # (1,1,Q)

        # signed difference and absolute value
        z   = (h.unsqueeze(1) - h.unsqueeze(0)).unsqueeze(-1)   # (M,M,1)
        r   = z.abs()                                             # (M,M,1)
        alpha = 5.0**0.5 / rho   # (1,1,Q)
        ar    = alpha * r         # (M,M,Q)  α·|z|
        E     = torch.exp(-ar)   # exp(−α|z|)

        # Matérn-5/2 envelope and cos/sin factors
        G   = (1.0 + ar + ar**2 / 3.0) * E          # γ,  (M,M,Q)
        C   = torch.cos(2.0 * np.pi * mu * z)        # (M,M,Q)
        S   = torch.sin(2.0 * np.pi * mu * z)        # (M,M,Q)
        m   = 2.0 * np.pi * mu                       # 2πμ, (1,1,Q)

        K   = torch.sum(w * G * C, dim=-1)           # (M,M)
        if order < 1:
            return K, None, None, None

        # dγ/dz = −(α²/3) · z · (1+αr) · E
        cm  = alpha**2 / 3.0                         # α²/3
        dG  = -cm * z * (1.0 + ar) * E              # (M,M,Q)

        dK  = torch.sum(w * (dG*C - m*G*S), dim=-1)
        if order < 2:
            return K, dK, None, None

        # d²γ/dz² = −(α²/3) · (1 + αr − α²r²) · E
        d2G = -cm * (1.0 + ar - ar**2) * E          # (M,M,Q)

        d2K = torch.sum(w * (d2G*C - 2.0*m*dG*S - m**2*G*C), dim=-1)
        if order < 3:
            return K, dK, d2K, None

        # d³γ/dz³ = (α⁴/3) · z · (3 − αr) · E
        d3G = (alpha**4 / 3.0) * z * (3.0 - ar) * E   # (M,M,Q)

        d3K = torch.sum(w * (d3G*C - 3.0*m*d2G*S - 3.0*m**2*dG*C + m**3*G*S), dim=-1)
        return K, dK, d2K, d3K

# ================================================================
# 3. GP-HM model  (Kronecker structure, same logic as original)
# ================================================================
class GPHM(nn.Module):
    def __init__(self, nx=100, ny=100, Q=30, F=100.0):
        super().__init__()
        self.nx, self.ny, self.M = nx, ny, nx*ny
        self.kx = StMixture1D(Q, F)
        self.ky = StMixture1D(Q, F)
        self.U        = nn.Parameter(torch.zeros(nx, ny))
        self.log_tau1 = nn.Parameter(torch.tensor(0.0))
        self.log_tau2 = nn.Parameter(torch.tensor(0.0))
        self.lambda_b = 500.0

    def forward_loss(self, pde_name, hx, hy, f_eval, u_true_bnd, bnd_mask, aux):
        order = 3 if pde_name == "KdV" else 2
        Kx, dKx, d2Kx, d3Kx = self.kx.get_K_dK_d2K_d3K(hx, order=order)
        Ky, dKy, d2Ky, _     = self.ky.get_K_dK_d2K_d3K(hy, order=2)

        Lx = safe_cholesky(Kx); Ly = safe_cholesky(Ky)

        # A = Cx⁻¹ U Cy⁻¹  (matrix form of C⁻¹ vec(U))
        A = torch.cholesky_solve(
                torch.cholesky_solve(self.U, Lx).T, Ly).T   # (nx,ny)

        u    = self.U
        u_x  = dKx  @ A @ Ky          # ∂u/∂x  on grid
        u_y  = Kx   @ A @ dKy.T       # ∂u/∂y  on grid  (dKy antisymmetric → .T = −dKy)
        u_xx = d2Kx @ A @ Ky          # ∂²u/∂x²
        u_yy = Kx   @ A @ d2Ky.T      # ∂²u/∂y² (d2Ky symmetric → .T = d2Ky)
        u_xxx = d3Kx @ A @ Ky if pde_name == "KdV" else None

        # PDE residual H
        if pde_name in ("Poisson", "HighFreq"):
            H = -(u_xx + u_yy) - f_eval
        elif pde_name == "Helmholtz":
            H = -(u_xx + u_yy) - _HELMHOLTZ_K2*u - f_eval
        elif pde_name == "VarCoeff":
            a, ax, ay = aux
            H = -(a*(u_xx+u_yy) + ax*u_x + ay*u_y) - f_eval
        elif pde_name == "SineGordon":
            H = u_yy - u_xx + torch.sin(u) - f_eval
        elif pde_name == "KdV":
            H = u_y + 6*u*u_x + u_xxx - f_eval
        else:
            raise ValueError(pde_name)

        # Prior
        logdet_C   = self.ny * 2*torch.sum(torch.log(torch.diag(Lx))) \
                   + self.nx * 2*torch.sum(torch.log(torch.diag(Ly)))
        loss_prior = 0.5*logdet_C + 0.5*torch.sum(self.U * A)

        # Boundary likelihood
        err_bnd  = self.U[bnd_mask] - u_true_bnd
        tau1     = torch.clamp(torch.exp(self.log_tau1), max=1e8)
        loss_bnd = self.lambda_b * (
            -0.5*err_bnd.numel()*self.log_tau1 + 0.5*tau1*torch.sum(err_bnd**2))

        # Equation likelihood
        tau2     = torch.clamp(torch.exp(self.log_tau2), max=1e8)
        loss_eq  = -0.5*self.M*self.log_tau2 + 0.5*tau2*torch.sum(H**2)

        return loss_prior + loss_bnd + loss_eq, H.detach().pow(2).mean(), err_bnd.detach().pow(2).mean()

    @torch.no_grad()
    def predict(self, xt: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor) -> torch.Tensor:
        """GP conditional mean at test points xt (N×2)."""
        Kx, _, _, _ = self.kx.get_K_dK_d2K_d3K(hx, order=0)
        Ky, _, _, _ = self.ky.get_K_dK_d2K_d3K(hy, order=0)
        Lx = safe_cholesky(Kx); Ly = safe_cholesky(Ky)
        A  = torch.cholesky_solve(
                 torch.cholesky_solve(self.U, Lx).T, Ly).T   # (nx,ny)

        def k1d_eval(sm: StMixture1D, xs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            """Cross-kernel between test points xs (N,) and grid h (M,) → (N,M)."""
            w     = torch.exp(sm.log_w)    # (Q,)
            rho   = torch.exp(sm.log_rho)  # (Q,)
            mu    = sm.mu                  # (Q,)
            z     = (xs.unsqueeze(1) - h.unsqueeze(0)).unsqueeze(-1)   # (N,M,1)
            r     = z.abs()
            alpha = 5.0**0.5 / rho         # (Q,)
            ar    = alpha * r              # (N,M,Q)
            E     = torch.exp(-ar)
            G     = (1.0 + ar + ar**2/3.0) * E
            C     = torch.cos(2.0*np.pi*mu*z)
            return torch.sum(w * G * C, dim=-1)   # (N,M)

        Kx_s = k1d_eval(self.kx, xt[:,0], hx)   # (N,nx)
        Ky_s = k1d_eval(self.ky, xt[:,1], hy)   # (N,ny)
        B    = Kx_s @ A                           # (N,ny)
        return torch.sum(B * Ky_s, dim=1)         # (N,)

# ================================================================
# 4. PDE setup helper
# ================================================================
def _setup_pde(pde_name: str, nx: int, ny: int):
    hx = torch.linspace(0, 1, nx, device=device)
    hy = torch.linspace(0, 1, ny, device=device)
    X, Y = torch.meshgrid(hx, hy, indexing='ij')
    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()

    f_eval      = torch.tensor(PDE_REGISTRY[pde_name].f_fn(Xn, Yn), device=device)
    u_true_grid = torch.tensor(PDE_REGISTRY[pde_name].u_fn(Xn, Yn), device=device)

    aux = None
    if pde_name == "VarCoeff":
        aux = (torch.tensor(_a_vc (Xn,Yn), device=device),
               torch.tensor(_ax_vc(Xn,Yn), device=device),
               torch.tensor(_ay_vc(Xn,Yn), device=device))

    bnd = torch.zeros((nx,ny), dtype=torch.bool, device=device)
    bnd[0,:] = bnd[-1,:] = bnd[:,0] = bnd[:,-1] = True

    return hx, hy, f_eval, u_true_grid[bnd], bnd, aux

# ================================================================
# 5. Train + evaluate
# ================================================================
def solve_gphm(pde_name: str, seed: int,
               nx: int = 100, ny: int = 100,
               Q: int = 30, F: float = 100.0,
               max_steps: int = 800_000) -> Tuple[float, float]:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    model = GPHM(nx, ny, Q, F).to(device)
    hx, hy, f_eval, u_true_bnd, bnd_mask, aux = _setup_pde(pde_name, nx, ny)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for step in range(max_steps):
        opt.zero_grad()
        loss, mse_eq, mse_bd = model.forward_loss(
            pde_name, hx, hy, f_eval, u_true_bnd, bnd_mask, aux)
        loss.backward()
        opt.step()
        if step % 500 == 0 and (mse_eq + mse_bd).item() < 1e-6:
            break
    train_time = time.time() - t0

    xt_np  = sample_interior(10_000)
    xt     = torch.tensor(xt_np, device=device)
    u_pred = model.predict(xt, hx, hy).cpu().numpy()
    u_true = PDE_REGISTRY[pde_name].u_fn(xt_np[:,0], xt_np[:,1])
    rmse   = float(np.sqrt(np.mean((u_pred - u_true)**2)))
    return rmse, train_time

# ================================================================
# 6. F selection: quick probe with seed=42, 15k steps
# ================================================================
F_CANDIDATES = (20, 40, 100)
PROBE_STEPS  = 15_000

def select_best_F(pde_name: str) -> int:
    best_F, best_rmse = F_CANDIDATES[0], float('inf')
    for F in F_CANDIDATES:
        rmse, _ = solve_gphm(pde_name, seed=42, F=float(F), max_steps=PROBE_STEPS)
        print(f"    [probe] {pde_name:12s}  F={F:3d}  RMSE={rmse:.3e}")
        if rmse < best_rmse:
            best_rmse, best_F = rmse, F
    print(f"    → best F={best_F}  (RMSE={best_rmse:.3e})")
    return best_F

# ================================================================
# 7. Main
# ================================================================
if __name__ == "__main__":
    print(f"Device: {device}\nPDEs: {PDE_NAMES}\n")

    # --- Step 1: select F per PDE ---
    print("=== F selection (probe: seed=42, 15k steps) ===")
    best_F_map: Dict[str, int] = {}
    for pde in PDE_NAMES:
        best_F_map[pde] = select_best_F(pde)

    # --- Step 2: full runs ---
    SEEDS = range(42, 47)
    print(f"\n=== Full experiment (seeds {list(SEEDS)}) ===")
    results: Dict[str, list] = {pde: [] for pde in PDE_NAMES}
    times:   Dict[str, list] = {pde: [] for pde in PDE_NAMES}

    for pde in PDE_NAMES:
        F = float(best_F_map[pde])
        print(f"\n--- {pde}  (F={int(F)}) ---")
        for seed in SEEDS:
            rmse, t = solve_gphm(pde, seed, F=F)
            results[pde].append(rmse)
            times[pde].append(t)
            print(f"  Seed={seed} | RMSE={rmse:.3e} | Time={t:.1f}s")

    # --- Summary ---
    print("\n=== GP-HM-StM  Summary (100×100 grid) ===")
    for pde in PDE_NAMES:
        d   = np.array(results[pde])
        res = bootstrap((d,), np.mean, confidence_level=0.95,
                        method="percentile", n_resamples=10_000, random_state=0)
        ci  = res.confidence_interval.high / 2 - res.confidence_interval.low / 2
        print(f"[{pde:12s}]  F={best_F_map[pde]:3d}  "
              f"RMSE = {d.mean():.4e} ± {ci:.4e}  ")

    for pde in PDE_NAMES:
        d   = np.array(times[pde])
        res = bootstrap((d,), np.mean, confidence_level=0.95,
                        method="percentile", n_resamples=10_000, random_state=0)
        ci  = res.confidence_interval.high / 2 - res.confidence_interval.low / 2
        m = d.mean()
        print(f"[{pde:12s}]  F={best_F_map[pde]:3d}  "
              f"{m:.3f}s  ± {ci:.3f}s")