"""Nudged elastic band implementation using only the standard library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence
import math


@dataclass
class NEBResult:
    path: List[List[float]]
    energies: List[float]
    gradients: List[List[float]]
    iterations: int
    max_force: float


def run_neb(
    potential: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    start: Sequence[float],
    end: Sequence[float],
    n_images: int = 20,
    k_spring: float = 2.0,
    step_size: float = 0.02,
    max_iter: int = 400,
    force_tolerance: float = 1e-3,
) -> NEBResult:
    """Run a lightweight NEB optimisation returning the discrete path."""

    start = [float(start[0]), float(start[1])]
    end = [float(end[0]), float(end[1])]
    images: List[List[float]] = []
    for index in range(n_images):
        t = index / (n_images - 1)
        images.append([
            start[0] * (1 - t) + end[0] * t,
            start[1] * (1 - t) + end[1] * t,
        ])

    forces: List[List[float]] = [[0.0, 0.0] for _ in range(n_images)]

    for iteration in range(max_iter):
        max_force = 0.0
        for idx in range(1, n_images - 1):
            prev_image = images[idx - 1]
            next_image = images[idx + 1]
            tangent = [next_image[0] - prev_image[0], next_image[1] - prev_image[1]]
            tangent_norm = math.hypot(tangent[0], tangent[1])
            if tangent_norm == 0:
                tangent = [0.0, 0.0]
            else:
                tangent = [tangent[0] / tangent_norm, tangent[1] / tangent_norm]

            grad = list(gradient(images[idx]))
            grad_parallel_scalar = grad[0] * tangent[0] + grad[1] * tangent[1]
            grad_parallel = [grad_parallel_scalar * tangent[0], grad_parallel_scalar * tangent[1]]
            grad_perp = [grad[0] - grad_parallel[0], grad[1] - grad_parallel[1]]

            dist_forward = math.hypot(next_image[0] - images[idx][0], next_image[1] - images[idx][1])
            dist_backward = math.hypot(images[idx][0] - prev_image[0], images[idx][1] - prev_image[1])
            spring_force_mag = k_spring * (dist_forward - dist_backward)
            spring_force = [spring_force_mag * tangent[0], spring_force_mag * tangent[1]]

            total_force = [-grad_perp[0] + spring_force[0], -grad_perp[1] + spring_force[1]]
            forces[idx] = total_force
            max_force = max(max_force, math.hypot(total_force[0], total_force[1]))

        if max_force < force_tolerance:
            break

        for idx in range(1, n_images - 1):
            images[idx][0] += step_size * forces[idx][0]
            images[idx][1] += step_size * forces[idx][1]

    energies = [float(potential(image)) for image in images]
    gradients = [list(gradient(image)) for image in images]
    return NEBResult(images, energies, gradients, iteration + 1, max_force)
