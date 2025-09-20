"""Utilities for creating potential energy surfaces without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math


@dataclass
class Gaussian:
    """Parameters describing a 2D isotropic Gaussian potential."""

    amplitude: float
    sigma: float
    x0: float
    y0: float

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Gaussian":
        return cls(
            amplitude=float(mapping.get("amplitude", 0.0)),
            sigma=max(float(mapping.get("sigma", 1.0)), 1e-6),
            x0=float(mapping.get("x0", 0.0)),
            y0=float(mapping.get("y0", 0.0)),
        )


@dataclass
class CircularState:
    """A circular region that defines a metastable state."""

    x0: float
    y0: float
    radius: float

    @classmethod
    def from_mapping(cls, mapping: dict) -> "CircularState":
        radius = float(mapping.get("radius", 1.0))
        if radius <= 0:
            raise ValueError("State radius must be positive.")
        return cls(
            x0=float(mapping.get("x0", 0.0)),
            y0=float(mapping.get("y0", 0.0)),
            radius=radius,
        )


@dataclass
class GridSpec:
    """Description of the 2D grid on which the PES is evaluated."""

    minimum: float
    maximum: float
    resolution: int

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "GridSpec":
        if not mapping:
            return cls(-4.0, 4.0, 60)
        minimum = float(mapping.get("minimum", -4.0))
        maximum = float(mapping.get("maximum", 4.0))
        if maximum <= minimum:
            raise ValueError("Grid maximum must be greater than the minimum.")
        resolution = int(mapping.get("resolution", 60))
        resolution = max(resolution, 10)
        return cls(minimum, maximum, resolution)


def _to_gaussians(items: Iterable[dict]) -> List[Gaussian]:
    return [Gaussian.from_mapping(item) for item in items]


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def create_grid(spec: GridSpec) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """Create a mesh grid and return both the mesh and axis vectors."""

    x_axis = linspace(spec.minimum, spec.maximum, spec.resolution)
    y_axis = linspace(spec.minimum, spec.maximum, spec.resolution)
    grid_x = [[x_axis[col] for col in range(spec.resolution)] for _ in range(spec.resolution)]
    grid_y = [[y_axis[row] for _ in range(spec.resolution)] for row in range(spec.resolution)]
    return grid_x, grid_y, x_axis, y_axis


def evaluate_potential(
    grid_x: Sequence[Sequence[float]],
    grid_y: Sequence[Sequence[float]],
    gaussians: Sequence[Gaussian],
) -> List[List[float]]:
    rows = len(grid_x)
    cols = len(grid_x[0]) if rows else 0
    potential = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for gaussian in gaussians:
        sigma2 = gaussian.sigma * gaussian.sigma
        for row in range(rows):
            for col in range(cols):
                dx = grid_x[row][col] - gaussian.x0
                dy = grid_y[row][col] - gaussian.y0
                r2 = dx * dx + dy * dy
                potential[row][col] += gaussian.amplitude * math.exp(-0.5 * r2 / sigma2)
    return potential


def gradient_potential(
    grid_x: Sequence[Sequence[float]],
    grid_y: Sequence[Sequence[float]],
    gaussians: Sequence[Gaussian],
) -> Tuple[List[List[float]], List[List[float]]]:
    rows = len(grid_x)
    cols = len(grid_x[0]) if rows else 0
    gx = [[0.0 for _ in range(cols)] for _ in range(rows)]
    gy = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for gaussian in gaussians:
        sigma2 = gaussian.sigma * gaussian.sigma
        for row in range(rows):
            for col in range(cols):
                dx = grid_x[row][col] - gaussian.x0
                dy = grid_y[row][col] - gaussian.y0
                r2 = dx * dx + dy * dy
                coeff = gaussian.amplitude * math.exp(-0.5 * r2 / sigma2) / sigma2
                gx[row][col] += -dx * coeff
                gy[row][col] += -dy * coeff
    return gx, gy


def evaluate_potential_at_point(point: Sequence[float], gaussians: Sequence[Gaussian]) -> float:
    x, y = point
    value = 0.0
    for gaussian in gaussians:
        dx = x - gaussian.x0
        dy = y - gaussian.y0
        sigma2 = gaussian.sigma * gaussian.sigma
        r2 = dx * dx + dy * dy
        value += gaussian.amplitude * math.exp(-0.5 * r2 / sigma2)
    return value


def evaluate_gradient_at_point(point: Sequence[float], gaussians: Sequence[Gaussian]) -> List[float]:
    x, y = point
    gx = 0.0
    gy = 0.0
    for gaussian in gaussians:
        dx = x - gaussian.x0
        dy = y - gaussian.y0
        sigma2 = gaussian.sigma * gaussian.sigma
        r2 = dx * dx + dy * dy
        coeff = gaussian.amplitude * math.exp(-0.5 * r2 / sigma2) / sigma2
        gx += -dx * coeff
        gy += -dy * coeff
    return [gx, gy]


def state_masks(
    grid_x: Sequence[Sequence[float]],
    grid_y: Sequence[Sequence[float]],
    state_a: CircularState,
    state_b: CircularState,
) -> Tuple[List[List[bool]], List[List[bool]]]:
    rows = len(grid_x)
    cols = len(grid_x[0]) if rows else 0
    mask_a = [[False for _ in range(cols)] for _ in range(rows)]
    mask_b = [[False for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            x = grid_x[row][col]
            y = grid_y[row][col]
            mask_a[row][col] = (x - state_a.x0) ** 2 + (y - state_a.y0) ** 2 <= state_a.radius**2
            mask_b[row][col] = (x - state_b.x0) ** 2 + (y - state_b.y0) ** 2 <= state_b.radius**2
    return mask_a, mask_b


def transition_state_boundary(
    grid_x: Sequence[Sequence[float]],
    grid_y: Sequence[Sequence[float]],
    potential: Sequence[Sequence[float]],
    transition_energy: float,
    tolerance: float,
) -> List[List[float]]:
    rows = len(grid_x)
    cols = len(grid_x[0]) if rows else 0
    points: List[List[float]] = []
    for row in range(rows):
        for col in range(cols):
            if abs(potential[row][col] - transition_energy) <= tolerance:
                points.append([grid_x[row][col], grid_y[row][col]])
    return points


def default_configuration() -> dict:
    """Return a configuration with three Gaussian features and two states."""

    return {
        "grid": {"minimum": -4.0, "maximum": 4.0, "resolution": 60},
        "gaussians": [
            {"amplitude": 4.0, "sigma": 1.2, "x0": -1.5, "y0": 0.0},
            {"amplitude": -6.5, "sigma": 0.8, "x0": 0.0, "y0": 0.0},
            {"amplitude": 5.0, "sigma": 1.0, "x0": 1.8, "y0": 0.6},
        ],
        "stateA": {"x0": -2.5, "y0": 0.0, "radius": 0.6},
        "stateB": {"x0": 2.4, "y0": 0.4, "radius": 0.6},
    }


def parse_configuration(payload: dict) -> tuple[
    GridSpec, List[Gaussian], CircularState, CircularState
]:
    """Parse input payload into strongly typed objects."""

    grid = GridSpec.from_mapping(payload.get("grid"))
    gaussians = _to_gaussians(payload.get("gaussians", []))
    if not gaussians:
        gaussians = _to_gaussians(default_configuration()["gaussians"])
    state_a = CircularState.from_mapping(payload.get("stateA", {}))
    state_b = CircularState.from_mapping(payload.get("stateB", {}))
    return grid, gaussians, state_a, state_b


def grid_min_max(values: Sequence[Sequence[float]]) -> tuple[float, float]:
    min_val = float("inf")
    max_val = float("-inf")
    for row in values:
        for value in row:
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value
    if min_val == float("inf"):
        min_val = 0.0
        max_val = 0.0
    return min_val, max_val
