import pickle
from glob import glob

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from constants import *
from scipy import interpolate
from scipy.interpolate import splev, splrep
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

np.random.seed(42)
start_clip = 200
end_clip = 100

# todo: download atran scripts
# todo: fit wavelength solution stuff
# todo: plot the maps


def perform_pca(input_matrix, n_princ_comp, return_noplanet=False):
    """
    Perform PCA using SVD.

    SVD is written as A = USV^H, where ^H is the Hermitian operator.

    Inputs
    ------
        :input_matrix:
        :n_princ_comp: number of principle components to keep
    """
    u, singular_values, vh = np.linalg.svd(
        input_matrix, full_matrices=False
    )  # decompose

    if return_noplanet:
        s_high_variance = singular_values.copy()
        s_high_variance[n_princ_comp:] = 0.0  # keeping the high-variance terms here
        s_matrix = np.diag(s_high_variance)
        A_noplanet[j] = np.dot(u, np.dot(s_matrix, vh))

    singular_values[:n_princ_comp] = 0.0  # zero out the high-variance terms here
    s_matrix = np.diag(singular_values)
    arr_planet = np.dot(u, np.dot(s_matrix, vh))

    if return_noplanet:
        return arr_planet, A_noplanet
    return arr_planet


def calc_doppler_shift(template_wave, template_flux, v):
    """
    Doppler shifts a spectrum. Evaluates the flux at a different grid.

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
    delta_lam = template_wave * beta.value
    shifted_wave = template_wave + delta_lam
    shifted_flux = np.interp(shifted_wave, template_wave, template_flux)
    return shifted_flux


def calc_exposure_time(
    period=1.80988198,
    mstar=1.458,
    e=0.000,
    inc=89.623,
    mplanet=0.894,
    rstar=1.756,
    peri=0,
    b=0.027,
    R=45000,
    pix_per_res=3.3,
    plot=False,
):
    """
    Todo: use the newer version of exoplanet, where the orbits are installed elsewhere.

    todo: refactor this into separate functions, maybe?

    Inputs
    -------
    autofilled for WASP-76b and IGRINS.

    R: (int) resolution of spectrograph.
    pix_per_res: (float) pixels per resolution element


    Outputs
    -------

    min_time: minimum time during transit before lines start smearing across resolution elements.

    min_time_per_pixel: minimum time during transit before lines start smearing across a single pixel.
    dphase_per_exp: the amount of phase (values from 0 to 1) taken up by a single exposure, given above constraints.
    n_exp: number of exposures you can take during transit.

    """

    orbit = KeplerianOrbit(
        m_star=mstar,  # solar masses!
        r_star=rstar,  # solar radii!
        #     m_planet_units = u.jupiterMass,
        t0=0,  # reference transit at 0!
        period=period,
        ecc=e,
        b=b,
        omega=np.radians(peri),  # periastron, radians
        Omega=np.radians(peri),  # periastron, radians
        m_planet=m_planet * 0.0009543,
    )
    t = np.linspace(0, period * 1.1, 1000)  # days
    vel = np.array(theano.function([], orbit.get_relative_velocity(t))())

    z_vel = vel[2] * 695700 / 86400  # km / s

    phases = t / period
    if plot:
        plt.plot(phases, z_vel)

        plt.axvline(0.5, color="gray", label="Eclipse")
        plt.axvline(0.25, label="Quadrature (should be maximized)", color="teal")

        plt.axvline(0.75, label="Quadrature (should be maximized)", color="teal")

        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Radial velocity (km/s)")
    acceleration = (
        np.diff(z_vel) / np.diff((t * u.day).to(u.s).si.value) * u.km / (u.s**2)
    )

    if plot:
        plt.plot(phases[:-1], acceleration)

        plt.axvline(0.5, color="gray", label="Eclipse")
        plt.axvline(0.25, label="Quadrature (should be minimized)", color="teal")

        plt.axvline(0.75, label="Quadrature (should be minimized)", color="teal")
        plt.xlabel("Orbital phase")
        plt.ylabel("Radial acceleration (km/s^2)")

        # cool, this is the acceleration.

        # now want the pixel crossing time.

        plt.legend()

    # R = c / delta v
    # delta v = c / R

    delta_v = const.c / R
    delta_v = delta_v.to(u.km / u.s)
    res_crossing_time = abs(delta_v / acceleration).to(u.s)
    if plot:
        plt.figure()
        plt.plot(phases[:-1], res_crossing_time)
        plt.axvline(0.5, color="gray", label="Eclipse")
        plt.axvline(0.25, label="Quadrature (should be maximized)", color="teal")

        plt.axvline(0.75, label="Quadrature (should be maximized)", color="teal")

        plt.legend()
        plt.xlabel("Orbital phase")
        plt.ylabel("Resolution crossing time (s)")
        plt.yscale("log")

        plt.figure()
        plt.plot(phases[:-1], res_crossing_time)
        plt.legend()

        # plt.yscale('log')
        plt.ylim(820, 900)
        plt.xlabel("Orbital phase")
        plt.ylabel("Resolution crossing time (s)")

        plt.xlim(0.96, 1.041)

    # todo: generalize this!
    ingress = 0.96
    egress = 1.04
    during_transit = (phases[:-1] > ingress) & (phases[:-1] < egress)

    res_crossing_time_transit = res_crossing_time[during_transit]

    min_time = np.min(res_crossing_time_transit)

    min_time_per_pixel = min_time / pix_per_res
    period = period * u.day
    dphase_per_exp = (np.min(res_crossing_time_transit) / period).si
    transit_dur = 30  # degrees. todo: calculate.
    n_exp = transit_dur / dphase_per_exp

    # then query https://igrins-jj.firebaseapp.com/etc/simple:
    # this gives us, for the given exposure time, what the SNR is going to be.
    return min_time, min_time_per_pixel, dphase_per_exp, n_exp


# todo: own convolution
def convolve_planet_spectrum(
    planet_wave,
    planet_flux,
    resolution_model=250000,
    resolution_instrument=45000,
):
    """
    Convolves a planet with rotational and instrumental profile.
    """
    scale = 1
    vsini = 4.52
    ker_rot = get_rot_ker(vsini, planet_wave)  # rotation kernel
    Fp_conv_rot = np.convolve(
        planet_flux, ker_rot, mode="same"
    )  # convolving with rot kernel

    resolution_ratio = resolution_model / resolution_instrument

    xker = np.arange(41) - 20
    sigma = resolution_ratio / (
        2.0 * np.sqrt(2.0 * np.log(2.0))
    )  # 5.5 is FWHM of resolution element in model wavelength grid "coords"
    yker = np.exp(-0.5 * (xker / sigma) ** 2.0)  # making a gaussian
    yker /= yker.sum()  # normalize so that the convolution is mathmatically correct.
    planet_flux_conv = np.convolve(Fp_conv_rot, yker, mode="same") * scale

    return planet_flux_conv, yker


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
    rv_planet = v_sys_tot + Kp * np.sin(2.0 * np.pi * phases) * 1e3  # measured in m/s

    rv_star = (
        v_sys_tot - Kstar * np.sin(2.0 * np.pi * phases)
    ) * 1e3  # measured in m/s. note opposite sign!
    return rv_planet, rv_star


def get_star_spline(star_wave, star_flux, planet_wave, yker, smooth=True):
    """
    calculates the stellar spline. accounting for convolution and such.

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
        Fluxes of the star. Measured in W/m^2/micron. todo: check.
    """
    star_spline = splrep(star_wave, star_flux, s=0.0)

    star_flux = splev(
        planet_wave, star_spline, der=0
    )  # now on same wavelength grid as planet wave

    star_flux = np.convolve(star_flux, yker, mode="same")  # convolving star too

    if smooth:
        star_flux = gaussian_filter1d(star_flux, 200)

    return star_flux


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
                wl_grid, flux, doppler_shift
            )

    return flux_cube_model


def add_blaze_function(wl_cube_model, flux_cube_model, n_order, n_exp):
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

    with open("K_blaze_spectra.pic", "rb") as f:
        K_blaze_cube = pickle.load(f)

    with open("H_blaze_spectra.pic", "rb") as f:
        H_blaze_cube = pickle.load(f)

    n_orders_k = K_blaze_cube.shape[0]
    n_orders_h = H_blaze_cube.shape[0]
    # K first, then H
    K_blaze_cube = detrend_cube(K_blaze_cube, n_orders_k, n_exp)
    H_blaze_cube = detrend_cube(H_blaze_cube, n_orders_h, n_exp)

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


def detrend_cube(cube, n_order, n_exp):
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
    for order in tqdm(range(n_order)):
        for exp in range(n_exp):
            max_val = np.max(cube[order, exp])
            cube[order, exp] /= max_val
    return cube
