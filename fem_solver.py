"""FEM solver for 2D plane stress using Q4 elements on a uniform rectangular mesh."""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class FEMSetup:
    """Precomputed mesh data. Create once at startup and reuse every iteration."""
    Ke0: np.ndarray      # (8, 8) element stiffness matrix at unit Young's modulus
    dof_map: np.ndarray  # (nely*nelx, 8) global DOF indices per element
    nelx: int
    nely: int
    ndof: int


@dataclass
class FEMResult:
    """Output of solve_primal. Contains everything needed for adjoint solves and sensitivity."""
    u: np.ndarray          # full displacement vector (ndof,)
    free_dofs: np.ndarray  # unconstrained DOF indices
    K_factor: object       # factorized K_free — call K_factor(rhs) to back-substitute


def compute_Ke0(E0: float, nu: float) -> np.ndarray:
    """
    Q4 element stiffness matrix for a unit square under plane stress.
    Integrated with 2x2 Gauss quadrature.
    Local node order: [BL, BR, TR, TL] → ξ=(-1,+1,+1,-1), η=(-1,-1,+1,+1).
    DOF order per element: [ux0,uy0, ux1,uy1, ux2,uy2, ux3,uy3].
    """
    D = E0 / (1 - nu**2) * np.array([
        [1,  nu, 0          ],
        [nu, 1,  0          ],
        [0,  0,  (1 - nu)/2 ],
    ])

    gp = 1.0 / np.sqrt(3)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    Ke0 = np.zeros((8, 8))
    for xi, eta in gauss_pts:
        # Shape function derivatives w.r.t. natural coordinates
        dN_dxi  = np.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)]) / 4
        dN_deta = np.array([-(1 - xi),  -(1 + xi),   (1 + xi),  (1 - xi) ]) / 4

        # For a unit square: J = 0.5*I, det(J) = 0.25, J_inv scales derivatives by 2
        dN_dx = 2.0 * dN_dxi
        dN_dy = 2.0 * dN_deta

        # Strain-displacement matrix B (3×8)
        B = np.zeros((3, 8))
        for k in range(4):
            B[0, 2*k]     = dN_dx[k]
            B[1, 2*k + 1] = dN_dy[k]
            B[2, 2*k]     = dN_dy[k]
            B[2, 2*k + 1] = dN_dx[k]

        Ke0 += B.T @ D @ B * 0.25  # Gauss weight = 1 each, det(J) = 0.25

    return Ke0


def build_dof_map(nelx: int, nely: int) -> np.ndarray:
    """
    Build (nely*nelx, 8) array mapping element index → 8 global DOF indices.
    Element e = elrow*nelx + elcol (row-major, matches density_matrix.ravel()).
    Node indexing: node(row, col) = row*(nelx+1) + col, row 0 at top.
    Local node order per element: [BL, BR, TR, TL].
    """
    er, ec = np.meshgrid(np.arange(nely), np.arange(nelx), indexing='ij')
    er = er.ravel()
    ec = ec.ravel()

    n_bl = (er + 1) * (nelx + 1) + ec        # bottom-left
    n_br = (er + 1) * (nelx + 1) + ec + 1    # bottom-right
    n_tr = er * (nelx + 1) + ec + 1           # top-right
    n_tl = er * (nelx + 1) + ec               # top-left

    return np.column_stack([
        2*n_bl, 2*n_bl + 1,
        2*n_br, 2*n_br + 1,
        2*n_tr, 2*n_tr + 1,
        2*n_tl, 2*n_tl + 1,
    ])


def setup_fem(nelx: int, nely: int, E0: float, nu: float) -> FEMSetup:
    """Precompute all mesh data. Call once before the optimization loop."""
    return FEMSetup(
        Ke0=compute_Ke0(E0, nu),
        dof_map=build_dof_map(nelx, nely),
        nelx=nelx,
        nely=nely,
        ndof=2 * (nelx + 1) * (nely + 1),
    )


def _assemble_K(
    density_matrix: np.ndarray,
    fem_setup: FEMSetup,
    penal: float,
    E0: float,
    E_min: float,
) -> csc_matrix:
    """Assemble global stiffness matrix using vectorized COO scatter."""
    dof_map = fem_setup.dof_map
    Ke0 = fem_setup.Ke0
    ndof = fem_setup.ndof
    n_elem = fem_setup.nelx * fem_setup.nely

    E_e = E_min + density_matrix.ravel()**penal * (E0 - E_min)  # (n_elem,)

    # Each element contributes 8×8 = 64 entries to K.
    # rows[e, i*8+j] = dof_map[e, i]  →  np.repeat each DOF 8 times
    # cols[e, i*8+j] = dof_map[e, j]  →  np.tile the DOF pattern 8 times
    rows = np.repeat(dof_map, 8, axis=1)                         # (n_elem, 64)
    cols = np.tile(dof_map, (1, 8))                              # (n_elem, 64)
    vals = np.outer(E_e, Ke0.ravel())                            # (n_elem, 64)

    return csc_matrix(
        (vals.ravel(), (rows.ravel(), cols.ravel())),
        shape=(ndof, ndof),
    )


def solve_primal(
    density_matrix: np.ndarray,
    fem_setup: FEMSetup,
    penal: float,
    E0: float,
    E_min: float,
    force_x: np.ndarray,
    force_y: np.ndarray,
    constrained_dofs: np.ndarray,
) -> FEMResult:
    """
    Assemble K, apply BCs, factorize, and solve K u = f.

    The factorized K is stored in the returned FEMResult so adjoint solves
    (which share the same K) can reuse it without reassembly or refactorization.

    force_x, force_y: (nely+1, nelx+1) arrays of nodal forces.
    constrained_dofs: 1D array of DOF indices with prescribed zero displacement.
    """
    ndof = fem_setup.ndof

    f = np.zeros(ndof)
    f[0::2] = force_x.ravel()
    f[1::2] = force_y.ravel()

    K = _assemble_K(density_matrix, fem_setup, penal, E0, E_min)

    free_dofs = np.setdiff1d(np.arange(ndof), constrained_dofs)
    K_free = K[np.ix_(free_dofs, free_dofs)].tocsc()
    K_factor = factorized(K_free)

    u = np.zeros(ndof)
    u[free_dofs] = K_factor(f[free_dofs])

    return FEMResult(u=u, free_dofs=free_dofs, K_factor=K_factor)


def solve_adjoint(
    primal: FEMResult,
    loaded_dofs: np.ndarray,
    fem_setup: FEMSetup,
) -> np.ndarray:
    """
    Solve the adjoint system K λ = b, where b[j] = 2·u[j] for j in loaded_dofs.

    This is the gradient of J = ‖u^P‖² with respect to u, fed back through K.
    Reuses the factorized K from the primal solve — no reassembly or refactorization.

    loaded_dofs: global DOF indices at which forces are applied for this load case.
    Returns full adjoint vector λ of length ndof.
    """
    ndof = fem_setup.ndof
    b = np.zeros(ndof)
    b[loaded_dofs] = 2.0 * primal.u[loaded_dofs]

    lam = np.zeros(ndof)
    lam[primal.free_dofs] = primal.K_factor(b[primal.free_dofs])
    return lam


def compute_sensitivity(
    primal: FEMResult,
    adjoint: np.ndarray,
    density_matrix: np.ndarray,
    fem_setup: FEMSetup,
    penal: float,
    E0: float,
    E_min: float,
) -> np.ndarray:
    """
    Compute dJ/dρ_e = −p·ρ_e^(p−1)·(E0−E_min)·λ_eᵀ·Ke0·u_e for each element.
    Returns (nely, nelx) sensitivity array.
    """
    # Retrieve known data from the FEM setup
    dof_map = fem_setup.dof_map #Map, each pixel has 8 values for indexes of correspoinding DOFs
    Ke0 = fem_setup.Ke0 #Single element stiffness matrix at unit Young's modulus


    rho_flat = density_matrix.ravel() #Flatten density matrix to (nelem,)
    dE_drho = penal * rho_flat**(penal - 1) * (E0 - E_min)  # Derivative of SIMP, neccesary for density to correctly affect stiffness

    u_elem   = primal.u[dof_map]  # (n_elem, 8) element-wise displacement vectors from FEM
    lam_elem = adjoint[dof_map]   # (n_elem, 8) element-wise adjoint vectors from adjoint solve, black magic

    # λ_eᵀ Ke0 u_e computed for all elements at once, black magic formula for finding sensitivity
    sensitivity = np.einsum('ei,ij,ej->e', lam_elem, Ke0, u_elem)  # (n_elem,)
    sensitivity = (-dE_drho * sensitivity).reshape(fem_setup.nely, fem_setup.nelx)
    return sensitivity


def compute_objective(
    primal: FEMResult,
    loaded_dofs: np.ndarray,
    weight: float = 1.0,
) -> float:
    """‖u^P‖² · weight for a single load case."""
    return weight * float(np.sum(primal.u[loaded_dofs]**2))


def plot_fem_results(
    primal: FEMResult,
    density_matrix: np.ndarray,
    fem_setup: FEMSetup,
    E0: float,
    nu: float,
    sensitivity: np.ndarray = None,
) -> None:
    """
    Figure with up to four panels: density, displacement magnitude, Von Mises stress,
    and optionally sensitivity (pass the (nely, nelx) array from compute_sensitivity).
    Stress is computed at each element centroid using E0 (valid for solid-material tests).
    """
    nelx, nely = fem_setup.nelx, fem_setup.nely
    u = primal.u

    n_panels = 4 if sensitivity is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Displacement magnitude at every node → (nely+1, nelx+1)
    u_mag = np.sqrt(u[0::2]**2 + u[1::2]**2).reshape(nely + 1, nelx + 1)

    # Constitutive matrix
    D = E0 / (1 - nu**2) * np.array([
        [1,  nu, 0          ],
        [nu, 1,  0          ],
        [0,  0,  (1 - nu)/2 ],
    ])

    # B matrix at element centroid (ξ=η=0):
    #   dN_dx = [-0.5, 0.5, 0.5, -0.5]
    #   dN_dy = [-0.5, -0.5, 0.5, 0.5]
    B_c = np.zeros((3, 8))
    for k, (dx, dy) in enumerate(zip([-0.5, 0.5, 0.5, -0.5], [-0.5, -0.5, 0.5, 0.5])):
        B_c[0, 2*k]     = dx
        B_c[1, 2*k + 1] = dy
        B_c[2, 2*k]     = dy
        B_c[2, 2*k + 1] = dx

    u_elem = u[fem_setup.dof_map]     # (n_elem, 8)
    sigma  = (u_elem @ B_c.T) @ D.T  # (n_elem, 3): [σ_xx, σ_yy, τ_xy]
    sxx, syy, txy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*txy**2).reshape(nely, nelx)

    im0 = axes[0].imshow(density_matrix, cmap='gray_r', vmin=0, vmax=1, origin='upper')
    axes[0].set_title('Density')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y (row 0 = top)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_mag, cmap='viridis', origin='upper')
    axes[1].set_title('Displacement magnitude')
    axes[1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(vm, cmap='hot', origin='upper')
    axes[2].set_title('Von Mises stress')
    axes[2].set_xlabel('x')
    plt.colorbar(im2, ax=axes[2])

    if sensitivity is not None:
        abs_max = np.max(np.abs(sensitivity))
        im3 = axes[3].imshow(sensitivity, cmap='RdBu', origin='upper',
                             vmin=-abs_max, vmax=abs_max)
        axes[3].set_title('Sensitivity dJ/drho')
        axes[3].set_xlabel('x')
        plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.show()
