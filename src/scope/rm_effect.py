import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from numba import njit
from tqdm import tqdm

const_c = const.c.to("km/s").value


# @njit
def make_grid(n_r, n_theta):
    r_range = np.linspace(0, 1, n_r)  # in stellar radius units

    theta_range = np.linspace(0, 2 * np.pi, n_theta)  # in radians

    # fine just make it uniform theta grid and weight later on by dA.
    r_vals = np.tile(r_range, n_theta)
    theta_vals = np.repeat(theta_range, n_r)
    # define a new theta at each radius

    grid = np.zeros((n_theta * n_r, 2))  # not quite sure this is correct. return!
    # pdb.set_trace()
    grid[:, 1] = r_vals
    grid[:, 0] = theta_vals
    return grid


# @njit
def doppler_shift_grid(grid, flux, wavelengths, v_rot):
    """
    the grid is the (r, theta) values. takes in a 1D spectrum.
    """
    # yeah this is where things get funky lol
    spectrum_grid = np.zeros((grid.shape[0], len(wavelengths)))

    # for i, point in tqdm(enumerate(grid), desc='doppler shifting', total=grid.shape[0]):
    for i, point in enumerate(grid):
        theta, r = point
        x = r * np.cos(theta)

        # calculate the v_dopp at that point assuming rotational broadening
        v_dopp = (
            x * v_rot
        )  # x is normalized from 0 to 1. this is now in km/s. TODO: cjheck this

        delta_lambda = wavelengths * v_dopp / const_c

        # interpolate the spectrum onto the new wavelength grid
        spectrum_grid[i] = np.interp(wavelengths + delta_lambda, wavelengths, flux)
        # todo: make the array math line up. and be less confusing.

    # doppler shift based on the wavel
    return spectrum_grid


# @njit
def calc_planet_locations(phases, r_star, inc, lambda_misalign, a):
    """
    calculate the x and y positions of the planet on the stellar disk.

    the phase is at 0.0 when the planet is in front of the star. 0.5 when it's behind the star.


    phases are in units of orbital phase. not radians.

    we need

    Assumes a circular orbit, for now...
    :param phases:
    :param r_p:
    :param r_star:
    :param inc:
    :param lambda_misalign:
    :param kp:
    :return:
    """

    # todo: add the spatial extent of the planet. for now, just assume it's a point source.

    positions = np.zeros(
        (len(phases), 2)
    )  # r and theta positions! need to know the lambda misalignment though lol
    for i, phase in enumerate(phases):
        # calculate the x and y positions of the planet on the stellar disk
        x = a * np.sin(2 * np.pi * phase) / r_star
        y = np.cos(inc) * a * np.cos(2 * np.pi * phase) / r_star

        # calculate the r and theta positions
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # now rotate the theta position by the misalignment angle
        theta += lambda_misalign

        positions[i] = r, theta
    return positions


# @njit
def occult_grid(occulted_grid, grid, planet_locations, r_p):
    """
    take the grid and the planet locations and occult the planet. grid is SPATIAL grid.

    :param grid:
    :param planet_locations:
    :param r_p:
    :return:
    """
    grid = grid.copy()
    for planet_location in planet_locations:
        # calculate the radial distance from the planet
        r = np.sqrt(
            (grid[:, 0] - planet_location[0]) ** 2
            + (grid[:, 1] - planet_location[1]) ** 2
        )

        # find the points where the planet is
        planet_points = np.where(r < r_p)

        # set those points to zero
        occulted_grid[planet_points] = 0.0
    return occulted_grid


# areas below
"""
    for i, area in enumerate(grid):
        r, theta = point
        spectrum = spectrum_grid[i]
        cell_area = r ** 2 * np.pi / (grid.shape[0] * grid.shape[1])
        summed_spectrum += cell_area * spectrum
        """


# @njit
def sum_grid(occulted_grid, areas):
    # todo: calculate the areas ahead of time. then just multiply by the spectrum.
    """
    sum up the grid. normalize by area.

    grid is the r theta situation
    :param grid:
    :return:
    """
    # pdb.set_trace()
    # what if i summed a slice
    summed_spectrum = np.sum(np.dot((occulted_grid).T, areas))
    # summed_spectrum = np.zeros(spectrum_grid.shape[1])
    # # can probably just multiply and broadcaset
    # for i, area in enumerate(areas):
    #     spectrum = spectrum_grid[i]
    #     summed_spectrum += area * spectrum

    # now it's in flux units that have area. it's ok if it's off by a multiplicative factor, I thnk. todo: check this.
    return summed_spectrum


def calc_areas(grid):
    """
    calculate the area of each cell in the grid.
    :param grid:
    :return:
    """
    areas = np.zeros(grid.shape[0])
    for i, point in enumerate(grid):
        r, theta = point
        areas[i] = r**2 * np.pi / (grid.shape[0] * grid.shape[1])
    return areas


def make_stellar_disk(
    flux,
    wavelengths,
    v_rot,
    phases,
    r_star,
    inc,
    lambda_misalign,
    a,
    r_p,
    n_theta=50,
    n_r=50,
):
    """
    based on the x and y position of the planet on the disk, make the stellar spectrum.

    note: doesn't incorporate CLV variations if I just use a phoenix model.

    and if i use an empirical spectrum, I'm not blocking out CLV variations.

    take the average of all these approaches? lol

    v_rot is in km/s

    r_p in meters, and same with rstar and a
    """

    # step 1: make the stellar grid. it's in polar so we don't have to worry about edge clipping. careful about summing up the area, though.

    grid = make_grid(n_r, n_theta)

    # step 2: doppler shift onto a bunch of different mu values. ignores CLV variations.
    spectrum_grid = doppler_shift_grid(
        grid, flux, wavelengths, v_rot
    )  # spectrum as a function of r and theta

    # step 3: where is the planet on the grid?
    planet_locations = calc_planet_locations(
        phases, r_star, inc, lambda_misalign, a
    )  # r and theta positions of the planet as a function of exposure

    # loop over the next bit: occurs at every phase.
    summed_grids = np.zeros((len(phases), len(wavelengths)))
    # occulted_grid = np.ones(spectrum_grid.shape)

    areas = calc_areas(grid)

    for i, phase in enumerate(phases):
        occulted_grid = np.copy(spectrum_grid)

        # step 4: occult those spots on the grid. oh. yeah it's a mask. so just multiply by spectrum grid!
        occulted_grid = occult_grid(occulted_grid, grid, planet_locations, r_p / r_star)

        # step 5: add up the rest of the spectra, normalized by area!
        summed_grids[i] = sum_grid(occulted_grid, areas)

        # clean up memory. these should be pretty big arrays...
        # occulted_grid.fill(1.0)

    # step 6: what's that overcorrection factor?
    correction_factor = (
        summed_grids / flux
    )  # this is what you get when you divide flux(in_transit) / flux(out).

    # you'll want to divide your correction factor by the data.

    return summed_grids, correction_factor

    # can do this convolution style assuming the same stellar line profile...
    # frame it as an area calculation. total minus the little bit you're
    # but are phoenix models already broadened? nope, they're not!