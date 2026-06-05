"""Topology optimization using SIMP + Optimality Criteria (OC) update."""

import numpy as np
import matplotlib.pyplot as plt
from fem_solver import (
    setup_fem,
    solve_primal,
    solve_adjoint,
    compute_sensitivity,
    compute_objective,
)


def give_output(density, sensitivity, objective, volume_fraction, iteration):
    """
    Called after every iteration. Replace the body of this function to customize
    what is shown during the optimization.
    """
    plt.clf()
    plt.imshow(density, cmap='gray_r', vmin=0, vmax=1, origin='upper')
    plt.title(f'Iter {iteration}  |  J = {objective:.4e}  |  V = {volume_fraction:.3f}')
    plt.colorbar()
    plt.pause(0.01)


def _oc_update(rho, sens, objective_volume, preserve, obstacle, rho_min, move, eta):
    """
    Optimality Criteria density update with bisection on the volume Lagrange multiplier.

    Update rule per element:
        rho_new = clip(rho * B^eta,  rho ± move,  [rho_min, 1])
        where B = max(0, -dJ/drho) / lambda

    Bisects lambda until the volume constraint is exactly met:
        sum(rho_new) = objective_volume * n_elem
    """
    V_target = objective_volume * rho.size

    l1, l2 = 1e-9, 1e9
    while (l2 - l1) / (l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        B = np.maximum(0.0, -sens / lmid)
        rho_new = np.maximum(rho_min,
                  np.maximum(rho - move,
                  np.minimum(1.0,
                  np.minimum(rho + move, rho * B**eta))))
        rho_new[preserve == 1] = 1.0
        rho_new[obstacle == 1] = rho_min
        if np.sum(rho_new) > V_target:
            l1 = lmid
        else:
            l2 = lmid

    return rho_new


def optimize_topology(
    nelx,
    nely,
    constrained_nodes,     # (nely+1, nelx+1): 1 = node fully fixed (both DOFs)
    load_cases,            # list of dicts: {'force_x': array, 'force_y': array, 'weight': float}
    preserve_geometries,   # (nely, nelx): 1 = element always solid
    obstacle_geometries,   # (nely, nelx): 1 = element always void
    objective_volume,      # target volume fraction alpha in (0, 1]
    E0=1.0,
    nu=0.3,
    penal=3.0,
    E_min=1e-9,
    rho_min=1e-3,
    max_iter=100,
    tol=0.01,
    move=0.2,
    eta=0.5,
):
    """
    Minimize sum_i w_i * ||u^P_i||^2 subject to sum(rho_e) <= alpha * n_elem.

    Each load case is a dict with keys:
        force_x  -- (nely+1, nelx+1) nodal x-forces
        force_y  -- (nely+1, nelx+1) nodal y-forces
        weight   -- scalar w_i (default 1.0)

    Returns: density (nely, nelx), sensitivity (nely, nelx), objective J, volume fraction
    """
    fem_setup = setup_fem(nelx, nely, E0, nu)

    # Convert constrained node mask → constrained DOF indices (both x and y per node)
    node_indices = np.where(constrained_nodes.ravel() == 1)[0]
    constrained_dofs = np.sort(np.concatenate([2 * node_indices, 2 * node_indices + 1]))

    # Translates the force arrays into the DOF indicies
    loaded_dofs_list = []
    for lc in load_cases:
        x_dofs = np.where(lc['force_x'].ravel() != 0)[0] * 2
        y_dofs = np.where(lc['force_y'].ravel() != 0)[0] * 2 + 1
        loaded_dofs_list.append(np.concatenate([x_dofs, y_dofs]))

    # Initial density: uniform at target volume fraction
    rho = np.full((nely, nelx), objective_volume)
    rho[preserve_geometries == 1] = 1.0
    rho[obstacle_geometries == 1] = rho_min

    # Initialise outputs so give_output can be called before the first real iteration
    total_sens = np.zeros((nely, nelx)) # sensitivity
    total_obj = 0.0 # objective value
    volume_fraction = float(np.mean(rho)) # current volume fraction

    plt.ion()

    for iteration in range(1, max_iter + 1):
        rho_old = rho.copy()

        # ── FEM + adjoint + sensitivity ──────────────────────────────────────
        total_sens = np.zeros((nely, nelx))
        total_obj = 0.0

        for lc, loaded_dofs in zip(load_cases, loaded_dofs_list): # Loops over load cases
            primal = solve_primal(
                rho, fem_setup, penal, E0, E_min,
                lc['force_x'], lc['force_y'], constrained_dofs,
            ) # Solves K(rho) u = f for the primal displacements u
            lam = solve_adjoint(primal, loaded_dofs, fem_setup) # Solves K(rho) λ = -dJ/du for the adjoint variables λ
            sens = compute_sensitivity(primal, lam, rho, fem_setup, penal, E0, E_min) # Computes dJ/drho using the primal and adjoint solutions
            w = lc.get('weight', 1.0)
            total_sens += w * sens
            total_obj += compute_objective(primal, loaded_dofs, weight=w)

        # ── Searches different lagrange multipliers, and updates density using move, eta and lagrange multiplier to preserve volume ────────────────────────────────────────────────
        rho = _oc_update(
            rho, total_sens, objective_volume,
            preserve_geometries, obstacle_geometries, rho_min, move, eta,
        )

        volume_fraction = float(np.mean(rho))
        change = float(np.max(np.abs(rho - rho_old)))

        give_output(rho, total_sens, total_obj, volume_fraction, iteration)
        print(f"Iter {iteration:4d}  J = {total_obj:.6e}  V = {volume_fraction:.4f}  change = {change:.4f}")

        if change < tol:
            print(f"Converged at iteration {iteration}.")
            break

    plt.ioff()
    return rho, total_sens, total_obj, volume_fraction
