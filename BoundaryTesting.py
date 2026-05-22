import os
import numpy as np
import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from torch.func import vmap, jacfwd
from matplotlib.colors import LogNorm
try:
    from safetensors.torch import save_file as safetensors_save
    from safetensors.torch import load_file as safetensors_load
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False
    safetensors_save = None
    safetensors_load = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DTYPE = torch.float64
SOLVE_DTYPE = torch.float64
torch.set_default_dtype(TRAIN_DTYPE)

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

# ==========================================
# 1. Define the exact solution and its derivatives (used to generate f and boundary conditions).
# ==========================================
x_sym, y_sym = sym.symbols("x y")
u_sym = sym.sin(2 * sym.pi * x_sym) * sym.sin(2 * sym.pi * y_sym) + sym.exp(-x_sym - y_sym)
f_sym = -(sym.diff(u_sym, x_sym, 2) + sym.diff(u_sym, y_sym, 2))

ux_sym = sym.diff(u_sym, x_sym)
uy_sym = sym.diff(u_sym, y_sym)

u_exact_fn = sym.lambdify((x_sym, y_sym), u_sym, "numpy")
f_exact_fn = sym.lambdify((x_sym, y_sym), f_sym, "numpy")
ux_exact_fn = sym.lambdify((x_sym, y_sym), ux_sym, "numpy")
uy_exact_fn = sym.lambdify((x_sym, y_sym), uy_sym, "numpy")

# ==========================================
# 2. Sampling Functions: L-Shape and Annulus
# ==========================================
def sample_lshape(n_int=3000, n_bd=600):
    # L-shape: [-1, 1]^2 with the bottom-right corner [0, 1] × [-1, 0] removed.
    pts = []
    while len(pts) < n_int:
        p = np.random.uniform(-1, 1, (n_int, 2))
        mask = ~((p[:, 0] > 0) & (p[:, 1] < 0))
        pts.extend(p[mask])
    pts_int = np.array(pts)[:n_int]

    # Sample the boundary in segments and assign normal vectors.
    # 1. Left (x=-1, y in [-1,1]), Normal: (-1, 0) - Dirichlet
    # 2. Top (y=1, x in [-1,1]), Normal: (0, 1) - Dirichlet
    # 3. Bottom-Left (y=-1, x in [-1,0]), Normal: (0, -1) - Dirichlet
    # 4. Right-Top (x=1, y in [0,1]), Normal: (1, 0) - Neumann
    # 5. Inner-Top (y=0, x in [0,1]), Normal: (0, -1) - Neumann
    # 6. Inner-Left (x=0, y in [-1,0]), Normal: (1, 0) - Neumann
    
    t = np.linspace(-1, 1, n_bd // 6)
    t_half = np.linspace(0, 1, n_bd // 6)
    t_half_neg = np.linspace(-1, 0, n_bd // 6)

    bds = [
        (np.c_[np.full_like(t, -1), t], np.c_[np.full_like(t, -1), np.zeros_like(t)], "D"), # Left
        (np.c_[t, np.full_like(t, 1)], np.c_[np.zeros_like(t), np.full_like(t, 1)], "D"),   # Top
        (np.c_[t_half_neg, np.full_like(t_half_neg, -1)], np.c_[np.zeros_like(t_half_neg), np.full_like(t_half_neg, -1)], "D"), # Bot-Left
        (np.c_[np.full_like(t_half, 1), t_half], np.c_[np.full_like(t_half, 1), np.zeros_like(t_half)], "N"), # Right-Top
        (np.c_[t_half, np.zeros_like(t_half)], np.c_[np.zeros_like(t_half), np.full_like(t_half, -1)], "N"), # Inner-Top
        (np.c_[np.zeros_like(t_half_neg), t_half_neg], np.c_[np.full_like(t_half_neg, 1), np.zeros_like(t_half_neg)], "N")  # Inner-Left
    ]
    return pts_int, bds

def sample_annulus(n_int=3000, n_bd=300, r_in=0.5, r_out=1.0):
    pts = []
    while len(pts) < n_int:
        p = np.random.uniform(-r_out, r_out, (n_int, 2))
        r2 = p[:, 0]**2 + p[:, 1]**2
        mask = (r2 >= r_in**2) & (r2 <= r_out**2)
        pts.extend(p[mask])
    pts_int = np.array(pts)[:n_int]

    theta = np.linspace(0, 2*np.pi, n_bd)
    # Inner boundary (Dirichlet), normal points INWARDS towards origin
    in_pts = np.c_[r_in*np.cos(theta), r_in*np.sin(theta)]
    in_normals = -in_pts / r_in
    
    # Outer boundary (Robin), normal points OUTWARDS
    out_pts = np.c_[r_out*np.cos(theta), r_out*np.sin(theta)]
    out_normals = out_pts / r_out

    bds = [
        (in_pts, in_normals, "D"),
        (out_pts, out_normals, "R")
    ]
    return pts_int, bds

# ==========================================
# 3. Core Solver Logic (Supports Mixed Boundaries)
# ==========================================
def precompute_complex_bases(model, pts_int, bds):
    def get_phi(x): return model(x.unsqueeze(0)).squeeze(0)
    jac1 = jacfwd(get_phi)
    jac2 = jacfwd(jac1)
    
    H = model.d_hidden
    pts_t = torch.tensor(pts_int, device=device, dtype=SOLVE_DTYPE)
    Phi_xx = torch.empty((len(pts_int), H), dtype=SOLVE_DTYPE, device=device)
    Phi_yy = torch.empty((len(pts_int), H), dtype=SOLVE_DTYPE, device=device)
    
    for i in range(0, len(pts_int), 64):
        b = min(64, len(pts_int) - i)
        j2 = vmap(jac2)(pts_t[i : i + b])
        Phi_xx[i : i + b] = j2[..., 0, 0]
        Phi_yy[i : i + b] = j2[..., 1, 1]

    A_bd_list, b_bd_list = [], []
    for pts, normals, b_type in bds:
        pts_t_bd = torch.tensor(pts, device=device, dtype=SOLVE_DTYPE)
        n_t = torch.tensor(normals, device=device, dtype=SOLVE_DTYPE)
        
        Phi = torch.empty((len(pts), H), dtype=SOLVE_DTYPE, device=device)
        Phi_x = torch.empty((len(pts), H), dtype=SOLVE_DTYPE, device=device)
        Phi_y = torch.empty((len(pts), H), dtype=SOLVE_DTYPE, device=device)
        
        for i in range(0, len(pts), 64):
            b = min(64, len(pts) - i)
            with torch.no_grad():
                Phi[i:i+b] = vmap(get_phi)(pts_t_bd[i:i+b])
            j1 = vmap(jac1)(pts_t_bd[i:i+b])
            Phi_x[i:i+b] = j1[..., 0]
            Phi_y[i:i+b] = j1[..., 1]

        x_np, y_np = pts[:, 0], pts[:, 1]
        u_exact = torch.tensor(u_exact_fn(x_np, y_np).reshape(-1,1), device=device, dtype=SOLVE_DTYPE)
        
        if b_type == "D":
            # Dirichlet: u = g
            A_bd_list.append(Phi)
            b_bd_list.append(u_exact)
        elif b_type == "N":
            # Neumann: du/dn = h
            ux_ext = torch.tensor(ux_exact_fn(x_np, y_np).reshape(-1,1), device=device, dtype=SOLVE_DTYPE)
            uy_ext = torch.tensor(uy_exact_fn(x_np, y_np).reshape(-1,1), device=device, dtype=SOLVE_DTYPE)
            h_exact = ux_ext * n_t[:, 0:1] + uy_ext * n_t[:, 1:2]
            
            Phi_n = Phi_x * n_t[:, 0:1] + Phi_y * n_t[:, 1:2]
            A_bd_list.append(Phi_n)
            b_bd_list.append(h_exact)
        elif b_type == "R":
            # Robin: u + du/dn = g_r
            ux_ext = torch.tensor(ux_exact_fn(x_np, y_np).reshape(-1,1), device=device, dtype=SOLVE_DTYPE)
            uy_ext = torch.tensor(uy_exact_fn(x_np, y_np).reshape(-1,1), device=device, dtype=SOLVE_DTYPE)
            g_r_exact = u_exact + (ux_ext * n_t[:, 0:1] + uy_ext * n_t[:, 1:2])
            
            Phi_n = Phi_x * n_t[:, 0:1] + Phi_y * n_t[:, 1:2]
            A_bd_list.append(Phi + Phi_n)
            b_bd_list.append(g_r_exact)

    A_bd = torch.cat(A_bd_list, dim=0)
    b_bd = torch.cat(b_bd_list, dim=0)
    
    return Phi_xx, Phi_yy, A_bd, b_bd

def solve_complex_geometry(model, shape="lshape"):
    if shape == "lshape":
        pts_int, bds = sample_lshape()
    else:
        pts_int, bds = sample_annulus()

    Phi_xx, Phi_yy, A_bd, b_bd = precompute_complex_bases(model, pts_int, bds)
    
    f_in = torch.tensor(f_exact_fn(pts_int[:,0], pts_int[:,1]).reshape(-1,1), device=device, dtype=SOLVE_DTYPE)
    A_in = -(Phi_xx + Phi_yy)
    b_in = f_in

    scale_in = 100.0 / (torch.amax(torch.abs(A_in), dim=1, keepdim=True) + 1e-12)
    scale_bd = 100.0 / (torch.amax(torch.abs(A_bd), dim=1, keepdim=True) + 1e-12)
    
    A_sys = torch.cat([A_in * scale_in, A_bd * scale_bd], dim=0)
    b_sys = torch.cat([b_in * scale_in, b_bd * scale_bd], dim=0)

    # Least squares
    w = torch.linalg.lstsq(A_sys, b_sys, rcond=None).solution
    return w

# ==========================================
# 4. Plotting Code
# ==========================================
def plot_results(model, w, shape="lshape"):
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'figure.dpi': 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    })

    # ===== grid =====
    grid_size = 400
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    if shape == "lshape":
        mask = (X > 0) & (Y < 0)
    else:
        r2 = X**2 + Y**2
        mask = (r2 < 0.5**2) | (r2 > 1.0**2)

    pts = torch.tensor(
        np.c_[X.flatten(), Y.flatten()],
        device=device,
        dtype=SOLVE_DTYPE
    )

    # ===== inference =====
    with torch.no_grad():
        u_pred = (model(pts) @ w).cpu().numpy().reshape(grid_size, grid_size)

    u_exact = u_exact_fn(X, Y)

    u_exact[mask] = np.nan
    u_pred[mask] = np.nan
    error = np.abs(u_pred - u_exact)

    # ===== metrics =====
    valid_err = error[~np.isnan(error)]
    rmse = np.sqrt(np.mean(valid_err**2))
    print(f"[{shape.upper()}] RMSE = {rmse:.4e}")

    # ===== solution color range (shared) =====
    sol_all = np.concatenate([
        u_exact[~np.isnan(u_exact)],
        u_pred[~np.isnan(u_pred)]
    ])
    vmin, vmax = sol_all.min(), sol_all.max()

    # ===== error color range (log, contrast-enhanced) =====
    err_pos = valid_err[valid_err > 0]
    vmin_err = max(err_pos.min(), 1e-12)
    vmax_err = np.percentile(err_pos, 99.5)

    # ===== plotting =====
    fig, axs = plt.subplots(
        1, 3,
        figsize=(15, 4),
        gridspec_kw=dict(wspace=0.25)
    )

    # --- Exact ---
    im0 = axs[0].pcolormesh(
        X, Y, u_exact,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        shading="auto"
    )
    axs[0].set_title("Exact Solution")

    # --- Prediction ---
    im1 = axs[1].pcolormesh(
        X, Y, u_pred,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        shading="auto"
    )
    axs[1].set_title("MetaColloc Prediction")

    # --- Error ---
    im2 = axs[2].pcolormesh(
        X, Y, error,
        cmap="magma",
        norm=LogNorm(vmin=vmin_err, vmax=vmax_err),
        shading="auto"
    )
    axs[2].set_title("Absolute Error")

    # ===== shared colorbar for solution =====
    cbar_sol = fig.colorbar(
        im1,
        ax=axs[:2],
        fraction=0.046,
        pad=0.04
    )
    cbar_sol.set_label(r"$u(x,y)$")

    # ===== error colorbar =====
    cbar_err = fig.colorbar(
        im2,
        ax=axs[2],
        fraction=0.046,
        pad=0.04
    )
    cbar_err.set_label(r"$|u - \hat{u}|$")

    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")

    plt.savefig(
        f"metacolloc_{shape}_H512.pdf",
        bbox_inches="tight"
    )
    print(f"Saved metacolloc_{shape}_H512.pdf")


# ==========================================
# 5. Execution Test (Assuming you have defined the MetaColloc class here)
# ==========================================
if __name__ == "__main__":
    SEED = 42
    H = 512
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Loading pre-trained model H={H}...")
    model = MetaColloc(2, H).to(device)
    
    state_dict = safetensors_load(f"./checkpoints/best_model_d2_h{H}_seed{SEED}.safetensors")
    
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("\n--- Testing L-shape with Dirichlet + Neumann BC ---")
    w_lshape = solve_complex_geometry(model, shape="lshape")
    plot_results(model, w_lshape, shape="lshape")

    print("\n--- Testing Annulus with Dirichlet + Robin BC ---")
    w_annulus = solve_complex_geometry(model, shape="annulus")
    plot_results(model, w_annulus, shape="annulus")