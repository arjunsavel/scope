import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import scipy.constants as constants
import numpy as np
import astropy.constants as const
from tqdm import tqdm
from scipy.interpolate import splrep, splev
import astropy.io.fits as fits
from scipy import interpolate

from scipy.ndimage import gaussian_filter1d
from glob import glob
# from exoplanet.orbits.keplerian import (
#     KeplerianOrbit
# )

import numpy as np

# import theano.tensor as tt
# import theano
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
np.random.seed(42)
start_clip=200
end_clip=100

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


def get_rot_ker(vsini, wStar):
    """
    Broadens by a single velocity?


    """
    (nx,) = wStar.shape
    dRV = np.mean(2.0 * (wStar[1:] - wStar[0:-1]) / (wStar[1:] + wStar[0:-1])) * 2.998e5
    nker = 401
    hnker = (nker - 1) // 2
    rker = np.zeros(nker)
    for ii in range(nker):
        ik = ii - hnker
        x = ik * dRV / vsini
        if np.abs(x) < 1.0:
            y = np.sqrt(1 - x**2)
            rker[ii] = y
    rker /= rker.sum()

    return rker


def convolve_planet_spectrum(
    planet_wave, planet_flux, resolution_model=250000, resolution_instrument=45000
):
    """
    Convolves a planet with rotational and instrumental profile.
    """
    scale = 1
    vsini = 4.52
    ker_rot = get_rot_ker(vsini, planet_wave)  # rotation kernel
    Fp_conv_rot = np.convolve(
        planet_flux, ker_rot, mode="same"
    )  # convolving with rot kernal

    # setting up instrumental profile. Assumes a Gaussian that is some number of "model" pixels wide.
    # model is right now at R=250K.  IGRINS is at R~45K. We make gaussian that is R_model/R_IGRINS ~ 5.5

    resolution_ratio = resolution_model / resolution_instrument

    xker = np.arange(41) - 20
    sigma = resolution_ratio / (
        2.0 * np.sqrt(2.0 * np.log(2.0))
    )  # 5.5 is FWHM of resolution element in model wavelength grid "coords"
    yker = np.exp(-0.5 * (xker / sigma) ** 2.0)  # making a gaussian
    yker /= yker.sum()  # normaliize
    planet_flux_conv = (
        np.convolve(Fp_conv_rot, yker, mode="same") * scale
    )  # conovlve..oo, that's where the "scale factor" happens

    return planet_flux_conv, yker


def calc_rvs(RVSYS, KP_planet, RV_star, phases):
    """
    calculate radial velocities of planet and star.
    """
    rv_planet = RVSYS + KP_planet * np.sin(
        2 * (np.pi * phases)
    )  # planet's radial velocity offset. km/s.
    rv_star = RVSYS - RV_star * np.sin(
        2 * (np.pi * phases)
    )  # star's radial velocity offset. km/s.
    return rv_planet, rv_star



def get_star_spline(star_wave, star_flux, planet_wave, yker, smooth=True):
    """
    calculates the stellar spline. accounting for convolution and such.
    """
    star_spline = interpolate.splrep(star_wave, star_flux, s=0.0)

    star_flux = splev(
        planet_wave, star_spline, der=0
    )  # now on same wavelength grid as planet wave

    star_flux = np.convolve(star_flux, yker, mode="same")  # convoving star too

    if smooth:
        star_flux = gaussian_filter1d(star_flux, 200)

    return star_flux



def read_atran(path):
    """
    reads a single file.
    """
    atran = pd.read_csv(
        path,
        index_col=0,
        names=["wav", "depth"],
        delim_whitespace=True,
    ).drop_duplicates(subset=["wav"])
    return atran


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
        dl_l = doppler_shift * 1E3 / constants.c

        # iterate through each exposure
        for order in range(n_order):
            order_flux = flux_cube_model[order][exp].copy()
            wl_grid = wl_cube_model[order]

            # do the Doppler shift
            shifted_wl_grid = wl_grid * (1.0 - dl_l)

            # then interpolate
            shifted_flux = np.interp(wl_grid, shifted_wl_grid, order_flux)
            flux_cube_model[order][exp] = shifted_flux

    return flux_cube_model


def read_atran_pair(path1, path2):
    """
    reads and concatenates an ATRAN file pair.
    """
    atran_40_first_half = read_atran(path1)

    atran_40_second_half = read_atran(path2)

    atran_40 = pd.concat([atran_40_first_half, atran_40_second_half]).drop_duplicates(
        subset=["wav"]
    )
    return atran_40

def find_closest_atran_angle(zenith_angle):
    """
    Parses through the available ATRAN files and finds the one with the zenith angle closest to the desired.
    """
    files = glob('data/atran*obslat*')
    zenith_paths = np.array([eval(file.split('_')[1]) for file in files])
    closest_ind = np.argmin(abs(zenith_paths - zenith_angle))
    closest_zenith = zenith_paths[closest_ind]
    return closest_zenith


def add_tellurics_atran(wl_cube_model, flux_cube_model, n_order, n_exp, vary_airmass=False):
    if vary_airmass:
        # assume now that I'm just using the same airmasses as before.
        zenith_angles = np.loadtxt('zenith_angles_w77ab.txt')

        # these are the beginning and ends of the two different ATRAN chunks.
        w_start_1 = 1.1
        w_end_1 = 2
        w_start_2 = 2
        w_end_2 = 3

        tell_splines = [] # will want one of these for each exposure.
        for zenith_angle in zenith_angles:
            zenith_path = find_closest_atran_angle(zenith_angle)
            path1 = f'data/atran_{zenith_path}_zenith_39_obslat_{w_start_1}_{w_end_1}_wave.dat'
            path2 = f'data/atran_{zenith_path}_zenith_39_obslat_{w_start_2}_{w_end_2}_wave.dat'
            atran_40 = read_atran_pair(path1, path2)
            tell_spline = interpolate.splrep(atran_40.wav, atran_40.depth, s=0.0)
            tell_splines += [tell_spline]

        for order in tqdm(range(n_order)):
            wl_grid = wl_cube_model[order]
            for exp in range(n_exp):
                tell_spline = tell_splines[exp]
                tell_flux = interpolate.splev(wl_grid, tell_spline, der=0)

                flux_cube_model[order, exp] *= tell_flux
    else:
        path1 = "atran_45k_1.3_2_microns_40_deg.txt"
        path2 = "atran_45k_2_2.7_microns_40_deg.txt"
        atran_40 = read_atran_pair(path1, path2)

        # todo: refactor. there are a lot of similarities!
        tell_spline = interpolate.splrep(atran_40.wav, atran_40.depth, s=0.0)
        for order in tqdm(range(n_order)):
            wl_grid = wl_cube_model[order]
            for exp in range(n_exp):
                tell_flux = interpolate.splev(wl_grid, tell_spline, der=0)
                flux_cube_model[order, exp] *= tell_flux

    return flux_cube_model


def calc_weighted_vector_insim(vals, eigenvector, relationships, provided_relationships=False):
    """
    Calculates the weighted vector for a given set of eigenvalues and eigenvectors.

    Inputs
    ------
        :vals: (array) eigenvalues
        :eigenvector: (array) eigenvectors
        :relationships: (list of arrays) relationships between eigenvectors and the flux vector.
        :provided_relationships: (bool) if True, then relationships are provided. If False, then relationships are calculated.

    Outputs
    -------
        :fm: (array) weighted vector
        :relationships: (list of arrays) relationships between eigenvectors and the flux vector.
    """

    fm = np.zeros(1698) # pad with 1s on all sides.
    if not provided_relationships:
        relationships = []
    for i, vector_ind in enumerate(range(4)):    
        relationship = relationships[i]

        vector = eigenvector[:,vector_ind]

        eigenweight = np.dot(relationship, vals)

        weighted_vector = eigenweight * vector

        fm += weighted_vector
        if not provided_relationships:
            relationships += [relationship]
    fm[fm < 0.] = 0.
    return fm, relationships

def eigenweight_func(airmass, date):
    """
    Does the weighting. turns date and airmass into their correct array!

    Inputs
    ------
        :airmass: (float) airmass
        :date: (float) date

    Outputs
    -------
        :array: (array) array of values to be multiplied by the eigenvectors.
    """
    if airmass < 1. or airmass > 3.:
        raise ValueError(f'Airmass must be valued between 1 amd 3. given airmass is {airmass}')
    if date < -200 or date > 400:
        raise ValueError
    return np.array([1, airmass, date, date**2])


def add_tellurics(wl_cube_model, flux_cube_model, n_order, n_exp, vary_airmass=False, tell_type='ATRAN',
                  time_dep_tell=False):
    """
    Includes tellurics in the model.
    todo: allow thew airmass variation to not be the case for the data-driven tellurics

    Inputs
    ------
        :wl_cube_model: (array) wavelength cube model
        :flux_cube_model: (array) flux cube model
        :n_order: (int) number of orders
        :n_exp: (int) number of exposures
        :vary_airmass: (bool) if True, then the airmass is varied. If False, then the airmass is fixed.
        :tell_type: (str) either 'ATRAN' or 'data-driven'
        :time_dep_tell: (bool) if True, then the tellurics are time dependent. If False, then the tellurics are not time dependent.

    Outputs
    -------
        :flux_cube_model: (array) flux cube model with tellurics included.

    """
    if tell_type == 'data-driven':
        eigenvectors = np.load('data/eigenvectors.npy')
        relationships_arr = np.load('data/eigenweight_coeffs.npy')
        wave_for_eigenvectors = np.load('data/wav_for_eigenvectors.npy')

        zenith_angles = np.loadtxt('zenith_angles_w77ab.txt')

        airmasses = 1/np.cos(np.radians(zenith_angles)) # todo: check units                                                                                                                                                                       

        if time_dep_tell:
            #dates = np.loadtxt('dates_w77ab_scaled.txt')
            dates = np.linspace(0, 350, len(airmasses))
        else:
            dates = np.ones_like(airmasses)


        # iterate through the each exposure.
        for i, airmass in enumerate(airmasses):
            date = dates[i]
            for order in range(relationships_arr.shape[0]):
                vals = eigenweight_func(airmass, date)
                fm, _ = calc_weighted_vector_insim(vals, eigenvectors[order], relationships_arr[order], provided_relationships=True)
                fm = np.concatenate([np.ones(150), fm])

                flux_cube_model[order][i] *= fm # the orders don't quite line up. or maybe they do : )                                                                                                                                            

    elif tell_type == 'ATRAN':
        flux_cube_model = add_tellurics_atran(wl_cube_model, flux_cube_model, n_order, n_exp, vary_airmass=vary_airmass)
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

    for order in tqdm(range(n_order), desc='adding blaze function'):
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

def add_constant_noise(flux_cube_model, wl_grid, SNR):
    """
    Adds constant noise to the flux cube model.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio

    Outputs
    -------
        :noisy_flux: (array) flux cube model with constant noise added.
    """
    n_photons = SNR ** 2
    # SNR**2 is the total number of photons.
    flux_cube = flux_cube_model.copy() * n_photons  # now scaled by number of photons. it's the maximum

    noise_matrix = np.random.normal(loc=0, scale=1, size=flux_cube_model.shape)
    noise_matrix_scaled = noise_matrix * np.sqrt(flux_cube)
    noisy_flux = flux_cube + noise_matrix_scaled
    return noisy_flux

def add_quadratic_noise(flux_cube_model, wl_grid, SNR, IGRINS=False, **kwargs):
    """
    Currently assumes that there are two bands: A and B.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio
        :IGRINS: (bool) if True, then the data is IGRINS data. If False, then the data is NOT IGRINS data.

    Outputs
    -------
        :noisy_flux: (array) flux cube model with quadratic noise added.
    """
    if IGRINS:
        A_a, A_b, A_c, B_a, B_b, B_c = np.loadtxt('igrins_median_snr.txt')
        # this is scaled to a mean SNR of 250.
        noisy_flux = np.ones_like(flux_cube_model) * 0
        wl_cutoff = 1.9
        medians = np.median(wl_grid, axis=1)
        
        A_wl = medians[medians < wl_cutoff]
        B_wl = medians[medians > wl_cutoff]

        A_SNRs = A_a + A_b * A_wl + A_c * A_wl ** 2
        B_SNRs = B_a + B_b * B_wl + B_c * B_wl ** 2

        # need to scale these to the required SNR.
        A_SNRs *= SNR / 250
        B_SNRs *= SNR / 250

        # iterate through each exposure
        for exposure in range(flux_cube_model.shape[1]):
            for order in range(flux_cube_model.shape[0]):
                if medians[order] < wl_cutoff: # first band
                    order_snr = A_SNRs[order - len(B_SNRs)]
                else: # second band
                    order_snr = B_SNRs[order]
                noisy_flux[order][exposure] = add_constant_noise(flux_cube_model[order][exposure], wl_grid, order_snr)

    else:
        raise NotImplementedError('Only IGRINS data is currently supported.')

    return noisy_flux


def add_igrins_noise(flux_cube_model, wl_grid, SNR):
    """
    Adds IGRINS noise to the flux cube model.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio

    Outputs
    -------
        :noisy_flux: (array) flux cube model with IGRINS noise added.
    """
    return add_quadratic_noise(flux_cube_model, wl_grid, SNR, IGRINS=True)

def add_custom_noise(SNR):
    """
    Adds custom noise to the flux cube model.
    """
    raise NotImplementedError

def add_noise_cube(flux_cube_model, wl_grid, SNR, noise_model='constant', **kwargs):
    """
    Per the equation in Brogi + Line 19.
    Assumes that the flux cube is scaled 0 to 1.

    I guess note that when I say "SNR", I'm ignoring for now the wavelength-dependent
    flux that you get because of a blaze function. It's the SNR at the peak of the blaze function.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio
        :noise_model: (str) noise model to use. Can be constant, IGRINS, custom_quadratic, or custom.

    Outputs
    -------
        :noisy_flux: (array) flux cube model with noise added.
    """

    noise_models = {'constant':add_constant_noise,
                        'IGRINS': add_igrins_noise,
                        'custom_quadratic':add_quadratic_noise,
                        'custom':add_custom_noise}
    if noise_model not in noise_models.keys():
        raise ValueError("Noise model can only be constant, IGRINS, custom_quadratic, or custom.")

    if np.max(flux_cube_model) > 2:  # generously
        raise ValueError(
            "Cannot add noise the way you want — this is not a normalized flux cube!"
        )


    noise_func = noise_models[noise_model]

    noisy_flux = noise_func(flux_cube_model, wl_grid, SNR, **kwargs)

    # need to make sure that it's still normed 0 to 1!
    noisy_flux = detrend_cube(noisy_flux, noisy_flux.shape[0], noisy_flux.shape[1])

    return noisy_flux


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