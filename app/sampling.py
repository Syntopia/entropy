"""Umbrella sampling utilities implemented without third-party dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence
import math
import random


@dataclass
class UmbrellaWindow:
    center_index: int
    reaction_coordinate: float
    free_energy: float
    samples: List[dict]


def umbrella_sampling(
    potential: Callable[[Sequence[float]], float],
    path: Sequence[Sequence[float]],
    beta: float = 1.0,
    spring_constant: float = 5.0,
    window_half_width: float = 0.5,
    samples_per_window: int = 120,
) -> List[UmbrellaWindow]:
    """Perform a lightweight umbrella sampling around the points of a path."""

    rng = random.Random(42)
    windows: List[UmbrellaWindow] = []
    total_images = len(path)

    for idx, point in enumerate(path):
        tangent = _estimate_tangent(path, idx)
        perp = [-tangent[1], tangent[0]]
        perp_norm = math.hypot(perp[0], perp[1])
        if perp_norm == 0:
            perp = [0.0, 1.0]
        else:
            perp = [perp[0] / perp_norm, perp[1] / perp_norm]

        samples = []
        weights = []
        energies = []
        for _ in range(samples_per_window):
            displacement_along = rng.gauss(0.0, window_half_width / 2.0)
            displacement_perp = rng.gauss(0.0, window_half_width)
            sample_point = [
                point[0] + displacement_along * tangent[0] + displacement_perp * perp[0],
                point[1] + displacement_along * tangent[1] + displacement_perp * perp[1],
            ]
            potential_energy = float(potential(sample_point))
            dx = sample_point[0] - point[0]
            dy = sample_point[1] - point[1]
            bias_energy = 0.5 * spring_constant * (dx * dx + dy * dy)
            weight = math.exp(-beta * (potential_energy + bias_energy))
            samples.append(
                {
                    "x": float(sample_point[0]),
                    "y": float(sample_point[1]),
                    "potential": potential_energy,
                    "bias": bias_energy,
                }
            )
            weights.append(weight)
            energies.append(potential_energy)

        partition = sum(weights)
        if partition == 0:
            free_energy = float("nan")
        else:
            expectation = sum(
                w * math.exp(-beta * e) for w, e in zip(weights, energies)
            ) / partition
            free_energy = -math.log(max(expectation, 1e-12)) / beta

        reaction_coordinate = idx / (total_images - 1) if total_images > 1 else 0.0
        windows.append(
            UmbrellaWindow(
                center_index=idx,
                reaction_coordinate=reaction_coordinate,
                free_energy=free_energy,
                samples=samples,
            )
        )

    return windows


def _estimate_tangent(path: Sequence[Sequence[float]], idx: int) -> List[float]:
    if idx == 0:
        tangent = [path[1][0] - path[0][0], path[1][1] - path[0][1]]
    elif idx == len(path) - 1:
        tangent = [path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]]
    else:
        tangent = [path[idx + 1][0] - path[idx - 1][0], path[idx + 1][1] - path[idx - 1][1]]
    norm = math.hypot(tangent[0], tangent[1])
    if norm == 0:
        return [1.0, 0.0]
    return [tangent[0] / norm, tangent[1] / norm]
