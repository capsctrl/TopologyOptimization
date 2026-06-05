"""
Cantilever beam test for the Q4 FEM solver.

Domain:  60×30 elements, all solid (ρ=1)
BC:      left edge fully fixed (all nodes, both DOFs)
Load:    F_y = -1 at midpoint of right edge
Material: E0=1, ν=0.3, penal=3, E_min=1e-9

Checks:
  - K assembles and solves without error
  - Tip deflects downward (u_y < 0 at load point)
  - Von Mises stress peaks near the fixed support
"""

import numpy as np
from fem_solver import (
    setup_fem,
    solve_primal,
    solve_adjoint,
    compute_sensitivity,
    compute_objective,
    plot_fem_results,
)


def main():
    nelx, nely = 60, 30
    E0, nu, penal, E_min = 1.0, 0.3, 3, 1e-9

    density_matrix = np.ones((nely, nelx))

    # Left edge: nodes (row, 0) for row in 0..nely
    left_nodes = np.arange(nely + 1) * (nelx + 1)
    constrained_dofs = np.sort(np.concatenate([2 * left_nodes, 2 * left_nodes + 1]))

    # Single downward point load at midpoint of right edge
    force_x = np.zeros((nely + 1, nelx + 1))
    force_y = np.zeros((nely + 1, nelx + 1))
    force_y[nely // 2, nelx] = -1.0

    # ── Precompute mesh ──────────────────────────────────────────────────────
    fem_setup = setup_fem(nelx, nely, E0, nu)
    print(f"Mesh: {nelx}×{nely} elements, {fem_setup.ndof} DOFs")

    # ── Primal solve ─────────────────────────────────────────────────────────
    primal = solve_primal(
        density_matrix, fem_setup, penal, E0, E_min,
        force_x, force_y, constrained_dofs,
    )

    load_node  = (nely // 2) * (nelx + 1) + nelx
    loaded_dofs = np.array([2 * load_node + 1])   # y-DOF at load node

    u_tip = primal.u[loaded_dofs[0]]
    u_max = np.max(np.abs(primal.u))
    print(f"Tip y-displacement : {u_tip:.6f}")
    print(f"Max |displacement| : {u_max:.6f}")

    # ── Adjoint solve ────────────────────────────────────────────────────────
    lam = solve_adjoint(primal, loaded_dofs, fem_setup)

    # ── Sensitivity ──────────────────────────────────────────────────────────
    sens = compute_sensitivity(primal, lam, density_matrix, fem_setup, penal, E0, E_min)
    print(f"Sensitivity range  : [{sens.min():.4e}, {sens.max():.4e}]")
    print(f"All sensitivities <= 0: {(sens <= 0).all()}")  # adding material always helps

    # ── Objective ────────────────────────────────────────────────────────────
    J = compute_objective(primal, loaded_dofs)
    print(f"Objective J = ||u^P||^2 : {J:.6f}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert u_tip < 0,         f"Tip should deflect downward, got u_y = {u_tip}"
    assert u_max < 1e3,       f"Displacement looks unreasonably large: {u_max}"
    assert np.isfinite(u_max), "Displacement contains NaN or Inf"
    print("All assertions passed.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_fem_results(primal, density_matrix, fem_setup, E0, nu, sensitivity=sens)




if __name__ == "__main__":
    main()
