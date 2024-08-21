import pytest
import os
from scipy import signal
from scipy.optimize import curve_fit

from scope.rm_effect import *

test_data_path = os.path.join(os.path.dirname(__file__), "../data")


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


@pytest.fixture
def test_load_phoenix_model():
    star_spectrum_path = os.path.join(test_data_path, "PHOENIX_5605_4.33.txt")
    star_wave, star_flux = np.loadtxt(
        star_spectrum_path
    ).T  # Phoenix stellar model packing
    star_wave *= 1e-6  # convert to meters
    return star_wave, star_flux


def test_doppler_shift_grid_shape(test_make_grid_values, test_load_phoenix_model):
    grid = test_make_grid_values
    star_wave, star_flux = test_load_phoenix_model
    n_wl = len(star_wave)

    res = doppler_shift_grid(grid, star_flux, star_wave, 3)
    assert res.shape == (grid.shape[0], n_wl)


def test_doppler_shift_grid_approach_side_more_blue(
    test_make_grid_values, test_load_phoenix_model
):
    grid = test_make_grid_values
    star_wave, star_flux = test_load_phoenix_model
    n_wl = len(star_wave)

    spectrum_grid = doppler_shift_grid(grid, star_flux, star_wave, 3)

    # from the approach side, the star should be more blue.
    max_r = np.max(grid[:, 0])
    most_right = grid[:, 1][np.argmin(np.abs(grid[:, 1] - 0))]
    most_left = grid[:, 1][np.argmin(np.abs(grid[:, 1] - np.pi))]

    # get the gridpoint that is most right and max r

    most_right_idx = np.argwhere((grid[:, 0] == max_r) & (grid[:, 1] == most_right))[0][
        0
    ]
    most_left_idx = np.argwhere((grid[:, 0] == max_r) & (grid[:, 1] == most_left))[0][0]

    # now get the spectra at these points from the spectrum grid.

    most_right_spec = spectrum_grid[most_right_idx]
    most_left_spec = spectrum_grid[most_left_idx]

    # now, the right one should be redshifted compared to the left one. how do we check this? simple max value!

    assert np.argmax(most_left_spec) < np.argmax(most_right_spec)


def test_doppler_shift_grid_approach_side_more_left_neg_vrot(
    test_make_grid_values, test_load_phoenix_model
):
    grid = test_make_grid_values
    star_wave, star_flux = test_load_phoenix_model
    n_wl = len(star_wave)

    spectrum_grid = doppler_shift_grid(grid, star_flux, star_wave, -3)

    # from the approach side, the star should be more blue.
    max_r = np.max(grid[:, 0])
    most_right = grid[:, 1][np.argmin(np.abs(grid[:, 1] - 0))]
    most_left = grid[:, 1][np.argmin(np.abs(grid[:, 1] - np.pi))]

    # get the gridpoint that is most right and max r

    most_right_idx = np.argwhere((grid[:, 0] == max_r) & (grid[:, 1] == most_right))[0][
        0
    ]
    most_left_idx = np.argwhere((grid[:, 0] == max_r) & (grid[:, 1] == most_left))[0][0]

    # now get the spectra at these points from the spectrum grid.

    most_right_spec = spectrum_grid[most_right_idx]
    most_left_spec = spectrum_grid[most_left_idx]

    # now, the right one should be redshifted compared to the left one. how do we check this? simple max value!

    assert np.argmax(most_left_spec) > np.argmax(most_right_spec)


@pytest.fixture()
def test_doppler_shift_grid_baseline(test_make_grid_values, test_load_phoenix_model):
    grid = test_make_grid_values
    star_wave, star_flux = test_load_phoenix_model
    n_wl = len(star_wave)

    spectrum_grid = doppler_shift_grid(grid, star_flux, star_wave, 5000)
    return spectrum_grid


def test_average_spectrum_not_input(
    test_load_phoenix_model, test_doppler_shift_grid_baseline
):
    star_wave, star_flux = test_load_phoenix_model
    spectrum_grid = test_doppler_shift_grid_baseline
    assert not np.allclose(star_flux, np.mean(spectrum_grid, axis=0))
