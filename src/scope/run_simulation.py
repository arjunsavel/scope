# import sys,os
# sys.path.append(os.path.realpath('..'))
# sys.path.append(os.path.realpath('.'))
import os
import sys

import jax
import numpy as np
from tqdm import tqdm

from scope.broadening import *
from scope.ccf import *
from scope.input_output import parse_input_file, write_input_file
from scope.noise import *
from scope.tellurics import *
from scope.utils import *

abs_path = os.path.dirname(__file__)

np.random.seed(42)


def make_data(
    scale,
    wlgrid,
    wl_model,
    Fp_conv,
    Fstar_conv,
    n_order,
    n_exposure,
    n_pixel,
    phases,
    Rp_solar,
    Rstar,
    do_pca=True,
    blaze=False,
    tellurics=False,
    n_princ_comp=4,
    v_sys=0.0,
    Kp=192.06,
    star=False,
    SNR=0,
    rv_semiamp_orbit=0.3229,
    observation="emission",
    tell_type="ATRAN",
    time_dep_tell=False,
    wav_error=False,
    order_dep_throughput=False,
    a=1,  # AU
    u1=0.3,
    u2=0.3,
    LD=True,
    b=0.0,  # impact parameter
    divide_out_of_transit=False,
    out_of_transit_dur=0.1,
    v_sys_measured=v_sys_measured,
):
    """
    Creates a simulated HRCCS dataset. Main function.

    Inputs
    ------
        :scale: (float) scaling factor for the data.
        :wlgrid: (array) wavelength grid for the data.
        :do_pca: (bool) if True, do PCA on the data.
        :blaze: (bool) if True, include the blaze function in the data.
        :tellurics: (bool) if True, include tellurics in the data.
        :n_princ_comp: (int) number of principal components to use.
        :v_sys: (float) systemic velocity of the planet.
        :Kp: (float) semi-amplitude of the planet.
        :star: (bool) if True, include the stellar spectrum in the data.
        :SNR: (float) photon signal-to-noise ratio (not of the planetary spectrum itself!)
        :observation: (str) 'emission' or 'transmission'.
        :tell_type: (str) 'ATRAN' or 'data-driven'.
        :time_dep_tell: (bool) if True, include time-dependent tellurics. not applicable for ATRAN.
        :wav_error: (bool) if True, include wavelength-dependent error.
        :order_dep_throughput: (bool) if True, include order-dependent throughput.

    Outputs
    -------
        :pca_noise_matrix: (array) the data cube with only the larger-scale trends.
        :flux_cube: (array) the data cube with the larger-scale trends removed.
        :fTemp_nopca: (array) the data cube with all components.
        :just_tellurics: (array) the telluric model that's multiplied to the dataset.

    """

    rv_planet, rv_star = calc_rvs(
        v_sys, v_sys_measured, Kp, rv_semiamp_orbit, phases
    )  # measured in m/s now

    flux_cube = np.zeros(
        (n_order, n_exposure, n_pixel)
    )  # will store planet and star signal
    pca_noise_matrix = np.zeros((n_order, n_exposure, n_pixel))
    for order in range(n_order):
        wlgrid_order = np.copy(wlgrid[order,])  # Cropped wavelengths
        flux_cube[order] = doppler_shift_planet_star(
            flux_cube[order],
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
        )

    throughput_baselines = np.loadtxt(abs_path + "/data/throughputs.txt")

    flux_cube = detrend_cube(flux_cube, n_order, n_exposure)

    for i in range(flux_cube.shape[1]):
        throughput_baseline = throughput_baselines[i]
        throughput_factor = throughput_baseline / np.nanpercentile(
            flux_cube[:, i, :], 90
        )
        flux_cube[:, i, :] *= throughput_factor

    if tellurics:
        flux_cube = add_tellurics(
            wlgrid,
            flux_cube,
            n_order,
            n_exposure,
            vary_airmass=True,
            tell_type=tell_type,
            time_dep_tell=time_dep_tell,
        )
        # these spline fits aren't perfect. we mask negative values to 0.
        just_tellurics = np.ones_like(flux_cube)
        just_tellurics = add_tellurics(
            wlgrid,
            just_tellurics,
            n_order,
            n_exposure,
            vary_airmass=True,
            tell_type=tell_type,
            time_dep_tell=time_dep_tell,
        )
        flux_cube[flux_cube < 0.0] = 0.0
        just_tellurics[just_tellurics < 0.0] = 0.0
    flux_cube = detrend_cube(flux_cube, n_order, n_exposure)
    if blaze:
        flux_cube = add_blaze_function(wlgrid, flux_cube, n_order, n_exposure)
        flux_cube[flux_cube < 0.0] = 0.0
    flux_cube[np.isnan(flux_cube)] = 0.0

    flux_cube = detrend_cube(flux_cube, n_order, n_exposure)
    flux_cube[np.isnan(flux_cube)] = 0.0
    if SNR > 0:  # 0 means don't add noise!
        if order_dep_throughput:
            noise_model = "IGRINS"
        else:
            noise_model = "constant"
        flux_cube = add_noise_cube(flux_cube, wlgrid, SNR, noise_model=noise_model)

    flux_cube = detrend_cube(flux_cube, n_order, n_exposure)

    if wav_error:
        doppler_shifts = np.loadtxt(
            "data/doppler_shifts_w77ab.txt"
        )  # todo: create this!
        flux_cube = change_wavelength_solution(wlgrid, flux_cube, doppler_shifts)

    flux_cube = detrend_cube(flux_cube, n_order, n_exposure)
    flux_cube[np.isnan(flux_cube)] = 0.0
    flux_cube_nopca = flux_cube.copy()

    if observation == "transmission" and divide_out_of_transit:
        # generate the out of transit baseline
        n_exposures_baseline = (
            out_of_transit_dur * n_exposure
        )  # assuming n_exposure is fully just in transit
        out_of_transit_flux = np.ones_like(
            flux_cube
        )  # todo: fix shape for out of transit baseline

        # take star
        for exposure in range(n_exposures_baseline):
            flux_star = calc_doppler_shift(
                wlgrid_order, wl_model, Fstar_conv, rv_star[exposure]
            )

            out_of_transit_flux[exposure,] *= flux_star

        out_of_transit_flux = detrend_cube(out_of_transit_flux, n_order, n_exposure)

        # add tellurics
        out_of_transit_flux = add_tellurics(
            wlgrid,
            out_of_transit_flux,
            n_order,
            n_exposures_baseline,
            vary_airmass=True,
            tell_type=tell_type,
            time_dep_tell=time_dep_tell,
        )
        out_of_transit_flux = detrend_cube(out_of_transit_flux, n_order, n_exposure)

        # add blaze
        out_of_transit_flux = add_blaze_function(
            wlgrid, out_of_transit_flux, n_order, n_exposures_baseline
        )
        out_of_transit_flux[flux_cube < 0.0] = 0.0
        out_of_transit_flux[np.isnan(flux_cube)] = 0.0
        out_of_transit_flux = detrend_cube(out_of_transit_flux, n_order, n_exposure)

        # add noise
        out_of_transit_flux = add_noise_cube(
            out_of_transit_flux, wlgrid, SNR, noise_model=noise_model
        )

        # take median
        median_out_of_transit = np.median(out_of_transit_flux, axis=1)

        # divide out the flux cube
        flux_cube /= median_out_of_transit  # todo: check axes work out

    if do_pca:
        for j in range(n_order):
            flux_cube[j] -= np.mean(flux_cube[j])
            flux_cube[j] /= np.std(flux_cube[j])
            flux_cube[j], pca_noise_matrix[j] = perform_pca(
                flux_cube[j], n_princ_comp, return_noplanet=True
            )
            # todo: think about the svd
            # todo: redo all analysis centering on 0?
    else:
        for j in range(n_order):
            for i in range(n_exposure):
                flux_cube[j][i] -= np.mean(flux_cube[j][i])

    if np.all(pca_noise_matrix == 0):
        print("was all zero")
        pca_noise_matrix = np.ones_like(pca_noise_matrix)
    if tellurics:
        return pca_noise_matrix, flux_cube, flux_cube_nopca, just_tellurics
    return (
        pca_noise_matrix,
        flux_cube,
        flux_cube_nopca,
        np.ones_like(flux_cube),
    )  # returning CCF and logL values


# @njit
def calc_log_likelihood(
    v_sys,
    Kp,
    scale,
    wlgrid,
    wl_model,
    Fp_conv,
    Fstar_conv,
    flux_cube,
    n_order,
    n_exposure,
    n_pixel,
    phases,
    Rp_solar,
    Rstar,
    rv_semiamp_orbit,
    A_noplanet,
    do_pca=True,
    n_princ_comp=4,
    star=False,
    observation="emission",
    a=1,  # AU
    u1=0.3,
    u2=0.3,
    LD=True,
    b=0.0,  # impact parameter
    v_sys_measured=0.0,
):
    """
    Calculates the log likelihood and cross-correlation function of the data given the model parameters.

    Inputs
    ------
          :v_sys: float
            Systemic velocity of the system in m/s
            :Kp: float
            Planet semi-amplitude in m/s
            :scale: float
            Planet-to-star flux ratio
            :wlgrid: array
            Wavelength grid
            :Fp_conv: array
            Planet spectrum convolved with the instrumental profile
            :Fstar_conv: array
            Stellar spectrum convolved with the instrumental profile
            :flux_cube: array
            Data cube
            :n_order: int
            Number of orders
            :n_exposure: int
            Number of exposures
            :n_pixel: int
            Number of pixels
            :phases: array
            Orbital phases
            :Rp_solar: float
            Planet radius in solar radii
            :Rstar: float
            Stellar radius in solar radii
            :rv_semiamp_orbit: float
            Semi-amplitude of the orbit in km/s. This is the motion of the *star*.
            :A_noplanet: array
            Array of the non-planet component of the data cube
            :do_pca: bool
            Whether to perform PCA on the data cube
            :n_princ_comp: int
            Number of principal components to use in the PCA
            :star: bool
            Whether to include the stellar component in the simulation.
            :observation: str
            Type of observation. Currently supported: 'emission', 'transmission'
    Outputs
    -------
            :logL: float
            Log likelihood of the data given the model parameters
            :ccf: array
            Cross-correlation function of the data given the model parameters
    """
    rv_planet, rv_star = calc_rvs(
        v_sys, v_sys_measured, Kp, rv_semiamp_orbit, phases
    )  # measured in m/s
    CCF = 0.0
    logL = 0.0
    for order in range(n_order):
        wlgrid_order = np.copy(wlgrid[order,])  # Cropped wavelengths
        model_flux_cube = np.zeros(
            (n_exposure, n_pixel)
        )  # "shifted" model spectra array at each phase

        model_flux_cube = doppler_shift_planet_star(
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
            reprocessing=True,
        )

        # ok now do the PCA. where does it fall apart?
        if do_pca:
            # process the model same as the "data"!
            model_flux_cube *= A_noplanet[order]
            model_flux_cube, _ = perform_pca(model_flux_cube, n_princ_comp, False)
        # I = np.ones(n_pixel)

        logl, ccf = calc_ccf(model_flux_cube, flux_cube[order], n_pixel)
        CCF += ccf
        logL += logl

        # # todo: airmass detrending reprocessing

    return logL, CCF  # returning CCF and logL values


def simulate_observation(
    planet_spectrum_path=".",
    star_spectrum_path=".",
    data_cube_path=".",
    phase_start=0,
    phase_end=1,
    n_exposures=10,
    observation="emission",
    blaze=True,
    n_princ_comp=4,
    star=True,
    SNR=250,
    telluric=True,
    tell_type="data-driven",
    time_dep_tell=False,
    wav_error=False,
    rv_semiamp_orbit=0.3229,
    order_dep_throughput=True,
    Rp=1.21,  # Jupiter radii,
    Rstar=0.955,  # solar radii
    kp=192.02,  # planetary orbital velocity, km/s
    v_rot=4.5,
    scale=1.0,
    v_sys=0.0,
    modelname="yourfirstsimulation",
    divide_out_of_transit=False,
    out_of_transit_dur=0.1,
    include_rm=False,
    v_rot_star=3.0,
    a=0.033,  #
    lambda_misalign=0.0,
    inc=90.0,
    **kwargs,
):
    """
    Run a simulation of the data, given a grid index and some paths. Side effects:
    writes some files in the output directory.

    Inputs
    ------
    grid_ind: int
        The index of the grid to use.
    planet_spectrum_path: str
        The path to the planet spectrum.
    star_spectrum_path: str
        The path to the star spectrum.
    phases: array-like
        The phases of the observations.
    observation: str
        Type of observation to simulate. Currently supported: emission, transmission.

    Outputs
    -------
    None

    """
    # make the output directory

    outdir = abs_path + f"/output/{modelname}"

    make_outdir(outdir)

    # and write the input file out
    output_args = locals()
    output_args.update(kwargs)
    write_input_file(output_args, output_file_path=f"{outdir}/input.txt")

    phases = np.linspace(phase_start, phase_end, n_exposures)
    Rp_solar = Rp * rjup_rsun  # convert from jupiter radii to solar radii
    Kp_array = np.linspace(kp - 100, kp + 100, 200)
    v_sys_array = np.arange(v_sys - 100, v_sys + 100)
    n_order, n_pixel = (44, 1848)  # todo: generalize.
    mike_wave, mike_cube = pickle.load(open(data_cube_path, "rb"), encoding="latin1")

    wl_cube_model = mike_wave.copy().astype(np.float64)

    wl_model, Fp, Fstar = np.load(planet_spectrum_path, allow_pickle=True)

    wl_model = wl_model.astype(np.float64)

    # Fp_conv_rot = broaden_spectrum(wl_model / 1e6, Fp, 0, vl=v_rot)
    rot_ker = get_rot_ker(v_rot, wl_model)
    Fp_conv_rot = np.convolve(Fp, rot_ker, mode="same")

    # instrument profile convolution
    instrument_kernel = get_instrument_kernel()
    Fp_conv = np.convolve(Fp_conv_rot, instrument_kernel, mode="same")

    star_wave, star_flux = np.loadtxt(
        star_spectrum_path
    ).T  # Phoenix stellar model packing

    if include_rm:
        star_flux, _ = make_stellar_disk(
            star_flux, star_wave, v_rot_star, phases, Rstar, inc, lambda_misalign, a, Rp
        )

    Fstar_conv = get_star_spline(
        star_wave, star_flux, wl_model, instrument_kernel, smooth=False
    )

    lls, ccfs = np.zeros((200, 200)), np.zeros((200, 200))

    # redoing the grid. how close does PCA get to a tellurics-free signal detection?
    A_noplanet, flux_cube, flux_cube_nopca, just_tellurics = make_data(
        scale,
        wl_cube_model,
        wl_model,
        Fp_conv,
        Fstar_conv,
        n_order,
        n_exposures,
        n_pixel,
        phases,
        Rp_solar,
        Rstar,
        do_pca=True,
        blaze=blaze,
        n_princ_comp=n_princ_comp,
        tellurics=telluric,
        v_sys=v_sys,
        star=star,
        Kp=kp,
        SNR=SNR,
        rv_semiamp_orbit=rv_semiamp_orbit,
        tell_type=tell_type,
        time_dep_tell=time_dep_tell,
        wav_error=wav_error,
        order_dep_throughput=order_dep_throughput,
        observation=observation,
        divide_out_of_transit=False,
        out_of_transit_dur=0.1,
        v_sys_measured=v_sys,
    )

    run_name = f"{n_princ_comp}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}"

    save_data(outdir, run_name, flux_cube, flux_cube_nopca, A_noplanet, just_tellurics)

    for l, Kp in tqdm(
        enumerate(Kp_array), total=len(Kp_array), desc="looping PCA over Kp"
    ):
        for k, v_sys in enumerate(v_sys_array):
            res = calc_log_likelihood(
                v_sys,
                Kp,
                scale,
                wl_cube_model,
                wl_model,
                Fp_conv,
                Fstar_conv,
                flux_cube,
                n_order,
                n_exposures,
                n_pixel,
                phases,
                Rp_solar,
                Rstar,
                rv_semiamp_orbit,
                do_pca=True,
                n_princ_comp=n_princ_comp,
                A_noplanet=A_noplanet,
                star=star,
                observation=observation,
                v_sys_measured=v_sys,
            )
            lls[l, k], ccfs[l, k] = res

    save_results(outdir, run_name, lls, ccfs)


if __name__ == "__main__":
    file = "input.txt"
    inputs = parse_input_file(file)
    simulate_observation(**inputs)
