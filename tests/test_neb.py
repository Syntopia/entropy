import math

from app.neb import run_neb
from app.pes import Gaussian


GAUSSIANS = [
    Gaussian(amplitude=-6.0, sigma=1.0, x0=0.0, y0=0.0),
    Gaussian(amplitude=3.0, sigma=0.5, x0=1.5, y0=0.0),
]


def potential(point):
    x, y = point
    value = 0.0
    for gaussian in GAUSSIANS:
        dx = x - gaussian.x0
        dy = y - gaussian.y0
        r2 = dx * dx + dy * dy
        sigma2 = gaussian.sigma * gaussian.sigma
        value += gaussian.amplitude * math.exp(-0.5 * r2 / sigma2)
    return value


def gradient(point):
    x, y = point
    gx = 0.0
    gy = 0.0
    for gaussian in GAUSSIANS:
        dx = x - gaussian.x0
        dy = y - gaussian.y0
        r2 = dx * dx + dy * dy
        sigma2 = gaussian.sigma * gaussian.sigma
        coeff = gaussian.amplitude * math.exp(-0.5 * r2 / sigma2) / sigma2
        gx += -dx * coeff
        gy += -dy * coeff
    return [gx, gy]


def test_neb_returns_path():
    result = run_neb(potential, gradient, start=(-2.0, 0.0), end=(2.0, 0.0), n_images=12, max_iter=50)
    assert len(result.path) == 12
    assert all(len(point) == 2 for point in result.path)
    assert all(math.isfinite(energy) for energy in result.energies)
    assert result.iterations <= 50
