"""
Utility functions for simulating HRCCS data.
"""

import json
import os
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from scope.constants import *

abs_path = os.path.dirname(__file__)

np.random.seed(42)
start_clip = 200
end_clip = 100


@njit
def doppler_shift_planet_star(
    model_flux_cube,
    n_exposure,
    phases,
    rv_planet,
    rv_star,
    wlgrid_order,
    wl_model,
    Fp_conv,
    Rp_solar,
    Fstar_conv,
    Rstar,
    u1,
    u2,
    a,
    b,
    LD,
    scale,
    star,
    observation,
    reprocessing=False,
):
    for exposure in range(n_exposure):
        _, flux_planet = calc_doppler_shift(
            wlgrid_order, wl_model, Fp_conv, rv_planet[exposure]
        )
        flux_planet *= scale  # apply scale factor

        _, flux_star = calc_doppler_shift(
            wlgrid_order, wl_model, Fstar_conv, rv_star[exposure]
        )

        if star and observation == "emission":
            model_flux_cube[exposure,] = (flux_planet * Rp_solar**2) / (
                flux_star * Rstar**2
            ) + 1.0
            if not reprocessing:
                model_flux_cube[exposure,] *= flux_star * Rstar**2
        elif observation == "emission":
            model_flux_cube[exposure,] = flux_planet * (Rp_solar * rsun) ** 2
        elif (
            observation == "transmission"
        ):  # in transmission, after we "divide out" (with PCA) the star and tellurics, we're left with Fp.
            I = calc_limb_darkening(u1, u2, a, b, Rstar, phases[exposure], LD)

            model_flux_cube[exposure,] = 1.0 - flux_planet * I

            if not reprocessing:
                model_flux_cube[exposure,] *= flux_star
    return model_flux_cube


def save_results(outdir, run_name, lls, ccfs):
    np.savetxt(
        f"{outdir}/lls_{run_name}.txt",
        lls,
    )
    np.savetxt(
        f"{outdir}/ccfs_{run_name}.txt",
        ccfs,
    )


def make_outdir(outdir):
    try:
        os.makedirs(outdir)
    except FileExistsError:
        print("Directory already exists. Continuing!")


def get_instrument_kernel(instrument, model_resolution=250000, kernel_size=41):
    """
    Creates a Gaussian kernel for instrument response using an alternative implementation.

    Parameters
    ----------
    resolution_ratio : float
        Ratio of resolutions (default: 250000/45000)
    kernel_size : int
        Size of the kernel (must be odd, default: 41)

    Returns
    -------
    numpy.ndarray
        Normalized Gaussian kernel
    """
    instrument_resolution_dict = {
        "IGRINS": 45000,
        "CRIRES+": 145000,
    }
    instrument_resolution = instrument_resolution_dict[instrument]
    resolution_ratio = instrument_resolution / model_resolution
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Convert resolution ratio to standard deviation
    std = resolution_ratio / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Create the Gaussian window
    gaussian_window = signal.windows.gaussian(
        kernel_size, std=std, sym=True  # Ensure symmetry
    )

    # Normalize to sum to 1
    normalized_kernel = gaussian_window / gaussian_window.sum()

    return normalized_kernel


def save_data(outdir, run_name, flux_cube, flux_cube_nopca, A_noplanet, just_tellurics):
    """
    Saves data to a pickle file.

    Inputs
    ------
        :outdir: (str) output directory
        :run_name: (str) name of the run
        :data: (dict) data to save
    """
    with open(
        f"{outdir}/simdata_{run_name}.txt",
        "wb",
    ) as f:
        pickle.dump(flux_cube, f)
    with open(
        f"{outdir}/nopca_simdata_{run_name}.txt",
        "wb",
    ) as f:
        pickle.dump(flux_cube_nopca, f)
    with open(
        f"{outdir}/A_noplanet_{run_name}.txt",
        "wb",
    ) as f:
        pickle.dump(A_noplanet, f)

    with open(f"{outdir}/just_tellurics_vary_airmass.txt", "wb") as f:
        pickle.dump(just_tellurics, f)


@njit
def calc_limb_darkening(u1, u2, a, b, Rstar, ph, LD):
    """
    calculates limb darkening as a function of phase

    :u1: (float) linear limb darkening coefficient
    :u2: (float) quadratic limb darkening coefficient
    :a: (float) semi-major axis in meters
    :b: (float) impact parameter
    :Rstar: (float) stellar radius in solar radii
    :ph: (array) phase
    :LD: (bool) whether to apply limb darkening
    """
    if LD:  # apply limb darkening. 1D style!
        x = (a * np.sin(2 * np.pi * ph)) / (Rstar * rsun)
        mu = np.sqrt(np.clip(1 - x**2 - b**2, 0, 1))
        if x**2 <= 1 - b**2:
            I = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
        else:
            I = 0.0
    else:
        I = 1.0
    return I


# todo: download atran scripts
# todo: fit wavelength solution stuff
# todo: plot the maps


@njit
def perform_pca(input_matrix, n_princ_comp, pca_removal="subtract"):
    """
    Perform PCA using SVD and remove principal components via subtraction or division.

    Parameters
    ----------
    input_matrix : ndarray
        2D array where rows are observations and columns are features.
    n_princ_comp : int
        Number of principal components to retain.
    mode : str, optional
        Method for removing PCA fit, either "subtract" or "divide".
        Default is "subtract".

    Returns
    -------
    corrected_data : ndarray
        The input matrix with PCA components removed.
    A_pca_fit : ndarray
        The PCA reconstruction used for removal.
    """
    # Perform SVD decomposition
    u, singular_values, vh = np.linalg.svd(input_matrix, full_matrices=False)

    # Construct PCA model of systematics
    s_high_variance = singular_values.copy()
    s_high_variance[n_princ_comp:] = 0.0  # Keep the high-variance trends
    s_matrix = np.diag(s_high_variance)
    A_pca_fit = np.dot(u, np.dot(s_matrix, vh))  # Systematics model

    # Apply chosen removal method
    if pca_removal == "subtract":
        corrected_data = input_matrix - A_pca_fit
    elif pca_removal == "divide":
        corrected_data = input_matrix / (A_pca_fit + 1e-8)  # Avoid division by zero
    else:
        raise ValueError("Mode must be 'subtract' or 'divide'.")

    return corrected_data, A_pca_fit


@njit
def calc_doppler_shift(eval_wave, template_wave, template_flux, v):
    """
    Doppler shifts a spectrum. Evaluates the flux at a different grid.
    convention: positive v is redshift.

    Inputs
    ------
        :wl: wavelength grid
        :flux: flux grid
        :v: velocity. Must be in m/s.

    Outputs
    -------
        :flux_shifted: shifted flux grid
    """
    beta = v / const_c
    delta_lam = eval_wave * beta
    shifted_wave = eval_wave - delta_lam
    shifted_flux = np.interp(shifted_wave, template_wave, template_flux)
    return shifted_wave, shifted_flux


@njit
def calc_rvs(v_sys, v_sys_measured, Kp, Kstar, phases):
    """
    calculate radial velocities of planet and star.

    Inputs
    ------
    v_sys: float
        Systemic velocity of the system. Measured in km/s
    v_sys_measured: float
        Measured systemic velocity of the system. Measured in km/s
    Kp: float
        Planet semi-amplitude. Measured in km/s
    Kstar: float
        Star semi-amplitude. Measured in km/s
    phases: array
        Orbital phases of the system. Measured in radians.

    Returns
    -------
    rv_planet: array
        Radial velocities of the planet. Measured in m/s
    rv_star: array
        Radial velocities of the star. Measured in m/s

    """
    v_sys_tot = v_sys + v_sys_measured  # total offset
    rv_planet = v_sys_tot + Kp * np.sin(
        2.0 * np.pi * phases
    )  # input in km/s, convert to m/s

    rv_star = v_sys_measured - Kstar * np.sin(
        2.0 * np.pi * phases
    )  # measured in m/s. note opposite sign!

    return rv_planet * 1e3, rv_star * 1e3


def get_star_spline(star_wave, star_flux, planet_wave, yker, smooth=True):
    """
    Calculates the stellar spline using an alternative implementation with linear interpolation
    instead of B-splines.

    Inputs
    ------
    star_wave: array
        Wavelengths of the star. Measured in microns.
    star_flux: array
        Fluxes of the star. Measured in W/m^2/micron.
    planet_wave: array
        Wavelengths of the planet. Measured in microns.
    yker: array
        Convolution kernel.
    smooth: bool
        Whether or not to smooth the star spectrum. Default is True.

    Returns
    -------
    star_flux: array
        Interpolated and processed stellar fluxes. Measured in W/m^2/micron.
    """
    # Create interpolation function using scipy's interp1d
    # Using cubic interpolation for smoother results
    interpolator = interp1d(
        star_wave,
        star_flux,
        kind="cubic",
        bounds_error=False,
        fill_value=(star_flux[0], star_flux[-1]),  # Extrapolate with edge values
    )

    # Interpolate onto planet wavelength grid
    interpolated_flux = interpolator(planet_wave)

    # Perform convolution
    convolved_flux = np.convolve(interpolated_flux, yker, mode="same")

    # Apply Gaussian smoothing if requested
    if smooth:
        final_flux = gaussian_filter1d(convolved_flux, sigma=200)
    else:
        final_flux = convolved_flux

    return final_flux


def change_wavelength_solution(wl_cube_model, flux_cube_model, doppler_shifts):
    """
    Takes a finalized wavelength cube and makes the wavelength solution for each exposure just slightly wrong.

    for now, it just shifts things. doesn't stretch.

    this basically interpolates the flux array onto a different wavelength grid. don't worry about the edge, it'll be
    trimmed?

    for now: uses prescribed pixel shifts!

    Inputs
    ------
        :wl_cube_model: (array) wavelength grid of simulated data!
        :flux_cube_model: (array) flux cube for simulated data!
        :pixel_shifts: (array) N_exp long array of number of wavelength pixels to shift. Convention: positive number is
                        redshift (shift to larger pixels). shifts must be in km/s.
    """
    n_order = flux_cube_model.shape[0]

    # iterate through each exposure
    for exp, doppler_shift in enumerate(doppler_shifts):
        doppler_shift *= 1e3  # convert to m/s

        # iterate through each exposure
        for order in range(n_order):
            wl_grid = wl_cube_model[order]
            flux = flux_cube_model[order][exp]
            flux_cube_model[order][exp] = calc_doppler_shift(
                wl_grid, wl_grid, flux, doppler_shift
            )

    return flux_cube_model


def add_blaze_function(
    wl_cube_model, flux_cube_model, n_order, n_exp, instrument="IGRINS"
):
    """
    Adds the blaze function to the model.

    Inputs
    ------
        :wl_cube_model: (array) wavelength cube model
        :flux_cube_model: (array) flux cube model
        :n_order: (int) number of orders
        :n_exp: (int) number of exposures

    Outputs
    -------
        :flux_cube_model: (array) flux cube model with blaze function included.
    """
    # read in...have to somehow match the telluric spectra
    if instrument != "IGRINS":
        raise NotImplementedError(
            "Only the IGRINS blaze function is currently supported."
        )

    with open(abs_path + "/data/K_blaze_spectra.pic", "rb") as f:
        K_blaze_cube = pickle.load(f)

    with open(abs_path + "/data/H_blaze_spectra.pic", "rb") as f:
        H_blaze_cube = pickle.load(f)

    n_orders_k = K_blaze_cube.shape[0]
    n_orders_h = H_blaze_cube.shape[0]
    # K first, then H
    K_blaze_cube = detrend_cube(K_blaze_cube, n_orders_k, n_exp, blaze=True)
    H_blaze_cube = detrend_cube(H_blaze_cube, n_orders_h, n_exp, blaze=True)

    for order in tqdm(range(n_order), desc="adding blaze function"):
        flux_cube_model_slice = flux_cube_model[order, :, :]
        if order >= n_orders_k:
            blaze_cube = H_blaze_cube
            order_used = order - 19
        else:
            blaze_cube = K_blaze_cube
            order_used = order
        flux_cube_model_slice = flux_cube_model_slice * blaze_cube[order_used][100:-100]
        flux_cube_model[order, :, :] = flux_cube_model_slice
    return flux_cube_model


def detrend_cube(cube, n_order, n_exp, blaze=False):
    """
    Detrends the cube by dividing each order by its maximum value.

    Inputs
    ------
        :cube: (array) flux cube
        :n_order: (int) number of orders
        :n_exp: (int) number of exposures

    Outputs
    -------
        :cube: (array) detrended flux cube
    """

    def detrend_array(array):
        max_val = np.nanmax(array)
        return array / max_val

    progress = tqdm(range(n_order), desc="detrending cube")

    if not blaze:
        for order in progress:
            for exp in range(n_exp):
                cube[order, exp] = detrend_array(cube[order, exp])

    else:
        for order in progress:
            cube[order] = detrend_array(cube[order])

    return cube


def unpack_grid(grid_ind, parameter_list):
    param_dict = parameter_list[grid_ind]

    (
        blaze,
        n_princ_comp,
        star,
        SNR,
        telluric,
        tell_type,
        time_dep_tell,
        wav_error,
        order_dep_throughput,
    ) = (
        param_dict["blaze"],
        param_dict["n_princ_comp"],
        param_dict["star"],
        param_dict["SNR"],
        param_dict["telluric"],
        param_dict["telluric_type"],
        param_dict["time_dep_telluric"],
        param_dict["wav_error"],
        param_dict["order_dep_throughput"],
    )
    return (
        blaze,
        n_princ_comp,
        star,
        SNR,
        telluric,
        tell_type,
        time_dep_tell,
        wav_error,
        order_dep_throughput,
    )
