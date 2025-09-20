from app.pes import (
    Gaussian,
    GridSpec,
    create_grid,
    evaluate_potential,
    gradient_potential,
    evaluate_gradient_at_point,
    transition_state_boundary,
)


def max_in_grid(values):
    return max(max(row) for row in values)


def test_gaussian_potential_symmetry():
    grid_spec = GridSpec(-2, 2, 20)
    grid_x, grid_y, _, _ = create_grid(grid_spec)
    gaussian = Gaussian(amplitude=5.0, sigma=0.5, x0=0.0, y0=0.0)
    potential = evaluate_potential(grid_x, grid_y, [gaussian])
    assert max_in_grid(potential) == potential[grid_spec.resolution // 2][grid_spec.resolution // 2]


def test_gradient_is_zero_at_center():
    gaussian = Gaussian(amplitude=5.0, sigma=0.5, x0=1.0, y0=-1.0)
    gradient = evaluate_gradient_at_point((1.0, -1.0), [gaussian])
    assert abs(gradient[0]) < 1e-8
    assert abs(gradient[1]) < 1e-8


def test_transition_state_boundary_extracts_points():
    grid_spec = GridSpec(-2, 2, 10)
    grid_x, grid_y, _, _ = create_grid(grid_spec)
    gaussian = Gaussian(amplitude=3.0, sigma=0.6, x0=0.0, y0=0.0)
    potential = evaluate_potential(grid_x, grid_y, [gaussian])
    points = transition_state_boundary(grid_x, grid_y, potential, transition_energy=2.0, tolerance=0.5)
    assert len(points) > 0
