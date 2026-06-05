"""
Topology optimization test cases.

1. test_cantilever  --  simple cantilever, no obstacles or preserved regions
2. test_bridge      --  simply supported bridge with a central void that
                        material must route around, and preserved support pads
"""

import numpy as np
import matplotlib.pyplot as plt
import topology_optimizer
from topology_optimizer import optimize_topology


# ── Shared visualization ─────────────────────────────────────────────────────

def make_give_output(preserve, obstacle):
    """
    Return a give_output function that draws the density map and overlays
    preserve regions (green) and obstacle regions (red) on top.
    Replaces the default give_output in topology_optimizer for each test.
    """
    def give_output(density, _, objective, volume_fraction, iteration):
       # LOVABLE - FILL IN WHATEVER CODE IS NECCESARY HERE TO VISUALIZE IN REAL TIME!

    return give_output


# ── Test 1: cantilever ───────────────────────────────────────────────────────

def test_cantilever():
    """60x30 cantilever, left edge fixed, point load at mid-right, 50% volume."""
    nelx, nely = 60, 30

    constrained_nodes = np.zeros((nely + 1, nelx + 1))
    constrained_nodes[:, 0] = 1                         # entire left edge fixed

    force_x = np.zeros((nely + 1, nelx + 1))
    force_y = np.zeros((nely + 1, nelx + 1))
    force_y[nely // 2, nelx] = -1.0                     # downward at mid-right

    load_cases = [{'force_x': force_x, 'force_y': force_y, 'weight': 1.0}]
    preserve = np.zeros((nely, nelx))
    obstacle = np.zeros((nely, nelx))

    topology_optimizer.give_output = make_give_output(preserve, obstacle)

    rho, sens, J, V = optimize_topology(
        nelx=nelx, nely=nely,
        constrained_nodes=constrained_nodes,
        load_cases=load_cases,
        preserve_geometries=preserve,
        obstacle_geometries=obstacle,
        objective_volume=0.5,
    )
    print(f"Cantilever  --  J = {J:.4e}  V = {V:.4f}")
    return rho


# ── Test 2: bridge with void ─────────────────────────────────────────────────

def test_bridge():
    """
    Simply supported bridge: 90x30, pinned at bottom corners, distributed
    downward load at top center.

    Obstacle:  central rectangle in the lower half of the domain — forces
               material to arch or truss around the opening.
    Preserve:  solid pads at the two support corners.
    Volume:    40% of total domain.
    """
    nelx, nely = 90, 30

    # Pin supports at the two bottom corners (both DOFs fixed)
    constrained_nodes = np.zeros((nely + 1, nelx + 1))
    constrained_nodes[nely, 0]    = 1
    constrained_nodes[nely, nelx] = 1

    # Distributed downward load across top-center third of the domain
    force_x = np.zeros((nely + 1, nelx + 1))
    force_y = np.zeros((nely + 1, nelx + 1))
    cx, span = nelx // 2, nelx // 6
    force_y[0, cx - span : cx + span + 1] = -1.0

    load_cases = [{'force_x': force_x, 'force_y': force_y, 'weight': 1.0}]

    # Obstacle: lower-half, central-third rectangle
    obstacle = np.zeros((nely, nelx))
    obstacle[nely // 2 :, nelx // 3 : 2 * nelx // 3] = 1

    # Preserve: 4-element solid pads at the two support corners
    preserve = np.zeros((nely, nelx))
    pad = 4
    preserve[nely - pad :, :pad]          = 1   # bottom-left pad
    preserve[nely - pad :, nelx - pad :]  = 1   # bottom-right pad

    topology_optimizer.give_output = make_give_output(preserve, obstacle)

    rho, sens, J, V = optimize_topology(
        nelx=nelx, nely=nely,
        constrained_nodes=constrained_nodes,
        load_cases=load_cases,
        preserve_geometries=preserve,
        obstacle_geometries=obstacle,
        objective_volume=0.4,
    )
    print(f"Bridge      --  J = {J:.4e}  V = {V:.4f}")
    return rho


# ── Test 3: multiple load cases ──────────────────────────────────────────────

def test_multi_load():
    """
    Cantilever under two load cases: downward force at top-right corner and
    downward force at bottom-right corner, equal weights.

    A single-load structure would be a diagonal bar pointing to one corner.
    With two load cases the optimizer must find a compromise — typically a
    branching or fan-shaped structure that serves both load paths at once.
    """
    nelx, nely = 60, 40

    constrained_nodes = np.zeros((nely + 1, nelx + 1))
    constrained_nodes[:, 0] = 1                          # left edge fixed

    # Load case 1: downward at top-right corner
    force_x1 = np.zeros((nely + 1, nelx + 1))
    force_y1 = np.zeros((nely + 1, nelx + 1))
    force_y1[0, nelx] = -1.0

    # Load case 2: downward at bottom-right corner
    force_x2 = np.zeros((nely + 1, nelx + 1))
    force_y2 = np.zeros((nely + 1, nelx + 1))
    force_y2[nely, nelx] = -1.0

    load_cases = [
        {'force_x': force_x1, 'force_y': force_y1, 'weight': 0.5},
        {'force_x': force_x2, 'force_y': force_y2, 'weight': 0.5},
    ]

    preserve = np.zeros((nely, nelx))
    obstacle = np.zeros((nely, nelx))

    topology_optimizer.give_output = make_give_output(preserve, obstacle)

    rho, sens, J, V = optimize_topology(
        nelx=nelx, nely=nely,
        constrained_nodes=constrained_nodes,
        load_cases=load_cases,
        preserve_geometries=preserve,
        obstacle_geometries=obstacle,
        objective_volume=0.4,
    )
    print(f"Multi-load  --  J = {J:.4e}  V = {V:.4f}")
    return rho


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Test 1: Cantilever ===")
    test_cantilever()

    print("\n=== Test 2: Bridge with void ===")
    test_bridge()

    print("\n=== Test 3: Multiple load cases ===")
    test_multi_load()

    plt.show()
