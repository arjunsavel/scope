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
