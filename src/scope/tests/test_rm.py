import pytest

from scope.rm_effect import *


@pytest.mark.parametrize(
    "values, output",
    [
        ([10, 10], (100, 2)),
        ([10, 1], (10, 2)),
        ([23, 2], (46, 2)),
    ],
)
def test_make_grid_shape(values, output):
    """
    Tests that the grid is the correct shape.
    """
    nr, ntheta = values
    grid = make_grid(nr, ntheta)

    assert grid.shape == output


@pytest.fixture
def test_make_grid_values():
    """
    Tests that the grid is the correct shape.
    """
    nr, ntheta = 10, 10
    grid = make_grid(nr, ntheta)

    return grid


def test_calc_areas_shape(test_make_grid_values):
    grid = test_make_grid_values
    areas = calc_areas(grid)
    assert areas.shape == (100,)


def test_calc_areas_all_nonzerro(test_make_grid_values):
    grid = test_make_grid_values
    areas = calc_areas(grid)
    assert np.all(areas > 0)


def test_calc_planet_locations_shape():
    phases = [1]
    r_star = 1
    a = 2
    inc = np.pi / 2
    lambda_misalign = 0
    res = calc_planet_locations(phases, r_star, inc, lambda_misalign, a)
    assert res.shape == (len(phases), 2)


@pytest.mark.parametrize(
    "values, output",
    [
        ([[0], 1, np.pi / 2, 0, 3], [[0, np.pi / 2]]),  # center of star at phase 0.
        (
            [[0], 1, np.pi / 2, 0, 3e3],
            [[0, np.pi / 2]],
        ),  # center of star at phase 0 even with a huge semimajor axis
        (
            [[0], 1, np.pi / 2, np.pi / 2, 3e3],
            [[0, np.pi]],
        ),  # center of star at phase 0 even with lambda misalignment
        (
            [[0], 1, np.pi / 2, np.pi / 2, 3e3],
            [[0, np.pi]],
        ),  # center of star at phase 0 even with lambda misalignment
        (
            [[0], 1, np.radians(85), np.pi / 2, 3e3],
            [[3e3 * np.cos(np.radians(85)), np.pi]],
        ),  # get a little bit of y.
        (
            [[0], 2, np.radians(85), np.pi / 2, 3e3],
            [[3e3 * np.cos(np.radians(85)) / 2, np.pi]],
        ),  # get a little bit of y, varying rstar.
        (
            [[0.5], 2, np.radians(85), np.pi / 2, 3e3],
            [[3e3 * np.cos(np.radians(85)) / 2, 0]],
        ),  # same answer at phase 0.5
        (
            [[0.5], 2, np.radians(85), 0, 3e3],
            [[3e3 * np.cos(np.radians(85)) / 2, -np.pi / 2]],
        ),  # and it flips if misaligntment woohoo
    ],
)
def test_calc_planet_locations_center_of_transit(values, output):
    # oh the positions are in r and theta lol
    phases, r_star, inc, lambda_misalign, a = values
    res = calc_planet_locations(phases, r_star, inc, lambda_misalign, a)
    assert np.allclose(res, output)
