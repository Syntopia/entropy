"""Core routines for constructing transition state theory visualisations."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from .neb import run_neb
from .pes import (
    CircularState,
    GridSpec,
    create_grid,
    default_configuration,
    evaluate_gradient_at_point,
    evaluate_potential,
    evaluate_potential_at_point,
    grid_min_max,
    parse_configuration,
    transition_state_boundary,
)
from .sampling import umbrella_sampling


def compute_visualisation(payload: Dict[str, Any]) -> Dict[str, Any]:
    grid_spec, gaussians, state_a, state_b = parse_configuration(payload)
    grid_x, grid_y, x_axis, y_axis = create_grid(grid_spec)
    potential_values = evaluate_potential(grid_x, grid_y, gaussians)

    def potential(point: Sequence[float]) -> float:
        return evaluate_potential_at_point(point, gaussians)

    def gradient(point: Sequence[float]):
        return evaluate_gradient_at_point(point, gaussians)

    start = _state_entry_point(grid_x, grid_y, potential_values, state_a)
    end = _state_entry_point(grid_x, grid_y, potential_values, state_b)

    neb_result = run_neb(
        potential,
        gradient,
        start,
        end,
        n_images=18,
        k_spring=2.5,
        step_size=0.03,
        max_iter=600,
        force_tolerance=5e-4,
    )

    transition_idx = max(range(len(neb_result.energies)), key=lambda i: neb_result.energies[i])
    transition_point = neb_result.path[transition_idx]
    transition_energy = float(neb_result.energies[transition_idx])

    boundary_points = transition_state_boundary(
        grid_x, grid_y, potential_values, transition_energy, tolerance=0.15
    )

    umbrellas = umbrella_sampling(potential, neb_result.path)
    z_min, z_max = grid_min_max(potential_values)

    response = {
        "grid": {
            "x": x_axis,
            "y": y_axis,
            "z": potential_values,
            "zMin": z_min,
            "zMax": z_max,
        },
        "states": {
            "A": {
                "x0": state_a.x0,
                "y0": state_a.y0,
                "radius": state_a.radius,
            },
            "B": {
                "x0": state_b.x0,
                "y0": state_b.y0,
                "radius": state_b.radius,
            },
        },
        "path": [
            {"x": point[0], "y": point[1], "z": energy}
            for point, energy in zip(neb_result.path, neb_result.energies)
        ],
        "transitionState": {
            "index": transition_idx,
            "x": transition_point[0],
            "y": transition_point[1],
            "energy": transition_energy,
        },
        "boundary": boundary_points,
        "umbrella": [
            {
                "index": window.center_index,
                "reactionCoordinate": window.reaction_coordinate,
                "freeEnergy": window.free_energy,
                "samples": window.samples,
            }
            for window in umbrellas
        ],
        "meta": {
            "nebIterations": neb_result.iterations,
            "nebMaxForce": neb_result.max_force,
        },
    }
    return response


def _state_entry_point(
    grid_x,
    grid_y,
    potential,
    state: CircularState,
) -> list[float]:
    rows = len(grid_x)
    cols = len(grid_x[0]) if rows else 0
    best_point = [state.x0, state.y0]
    best_energy = float("inf")
    for row in range(rows):
        for col in range(cols):
            x_val = grid_x[row][col]
            y_val = grid_y[row][col]
            if (x_val - state.x0) ** 2 + (y_val - state.y0) ** 2 <= state.radius**2:
                energy = potential[row][col]
                if energy < best_energy:
                    best_energy = energy
                    best_point = [x_val, y_val]
    return best_point


__all__ = [
    "compute_visualisation",
    "default_configuration",
]
