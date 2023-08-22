import sys

from broadening import *
from grid import *
from tellurics import *

from src.scope.ccf import *
from src.scope.noise import *
from src.scope.utils import *

np.random.seed(42)


def make_data(
    scale,
    wlgrid,
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
    do_airmass_detrending=False,
    tellurics=False,
    n_princ_comp=4,
    v_sys=0,
    Kp=192.06,
    star=False,
    SNR=0,
    observation="emission",
    tell_type="ATRAN",
    time_dep_tell=False,
    wav_error=False,
    order_dep_throughput=False,
):
    """
    Creates a simulated HRCCS dataset. Main function.

    Inputs
    ------
        :scale: (float) scaling factor for the data.
        :wlgrid: (array) wavelength grid for the data.
        :do_pca: (bool) if True, do PCA on the data.
        :blaze: (bool) if True, include the blaze function in the data.
        :do_airmass_detrending: (bool) if True, do airmass detrending on the data.
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
        :A_noplanet: (array) the data cube with only the larger-scale trends.
        :flux_cube: (array) the data cube with the larger-scale trends removed.
        :fTemp_nopca: (array) the data cube with all components.
        :just_tellurics: (array) the telluric model that's multiplied to the dataset.

    """
    Kstar = 0.3229 * 1.0

    v_sys_measured = (
        1.6845  # this is the systemic velocity of the system reported in the literature
    )

    rv_planet, rv_star = calc_rvs(
        v_sys, v_sys_measured, Kp, Kstar, phases
    )  # measured in m/s

    flux_cube = np.zeros(
        (n_order, n_exposure, n_pixel)
    )  # will store planet and star signal
    A_noplanet = np.zeros((n_order, n_exposure, n_pixel))
    for order in range(n_order):
        wlgrid_order = np.copy(wlgrid[order,])  # Cropped wavelengths
        for exposure in range(n_exposure):
            flux_planet = calc_doppler_shift(
                wlgrid_order, Fp_conv * Rp_solar**2, rv_planet[exposure]
            )
            flux_planet *= scale  # apply scale factor
            flux_star = calc_doppler_shift(
                wlgrid_order, Fstar_conv * Rstar**2, rv_star[exposure]
            )
        if star:
            if observation == "emission":
                flux_cube[
                    order,
                    exposure,
                ] = (
                    flux_planet + flux_star  # now want to get the star in there
                )
            elif observation == "transmission":
                flux_cube[
                    order,
                    exposure,
                ] = flux_star * (
                    1 - flux_planet
                )  # now want to get the star in there
            else:
                flux_cube[
                    order,
                    exposure,
                ] = flux_planet

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
    if blaze:
        flux_cube = add_blaze_function(wlgrid, flux_cube, n_order, n_exposure)
        flux_cube[flux_cube < 0.0] = 0.0
    flux_cube = detrend_cube(
        flux_cube, n_order, n_exposure
    )  # TODO: think about mean subtraction vs. division here? also, this is basically the wavelength-calibration step!
    if SNR > 0:  # 0 means don't add noise!
        if order_dep_throughput:
            noise_model = "IGRINS"
        else:
            noise_model = "constant"
        flux_cube = add_noise_cube(flux_cube, wlgrid, SNR, noise_model=noise_model)

    if wav_error:
        doppler_shifts = np.loadtxt(
            "data/doppler_shifts_w77ab.txt"
        )  # todo: create this!
        flux_cube = change_wavelength_solution(wlgrid, flux_cube, doppler_shifts)
    flux_cube_nopca = flux_cube.copy()
    if do_pca:
        for j in range(n_order):
            flux_cube[j], A_noplanet[j] = perform_pca(
                flux_cube, n_princ_comp, return_noplanet=True
            )
            # todo: think about the svd
            # todo: redo all analysis centering on 0?
    elif do_airmass_detrending:
        zenith_angles = np.loadtxt("data/zenith_angles_w77ab.txt")
        airm = 1 / np.cos(np.radians(zenith_angles))  # todo: check units
        polyDeg = 2.0  # Setting the degree of the polynomial fit
        xvec = airm
        fAirDetr = np.zeros((n_order, n_exposure, n_pixel))
        # todo: think about looping
        for io in range(n_order):
            # Now looping over columns
            for i in range(n_pixel):
                yvec = flux_cube[io, :, i].copy()
                fit = np.poly1d(np.polyfit(xvec, yvec, polyDeg))(xvec)
                fAirDetr[io, :, i] = flux_cube[io, :, i] / fit
                A_noplanet[io, :, i] = fit
        flux_cube = fAirDetr
        flux_cube[~np.isfinite(flux_cube)] = 0.0
    else:
        for j in range(n_order):
            for i in range(n_exposure):
                flux_cube[j][i] -= np.mean(flux_cube[j][i])

    # todo: check vars
    if np.all(A_noplanet == 0):
        print("was all zero")
        A_noplanet = np.ones_like(A_noplanet)
    if tellurics:
        return A_noplanet, flux_cube, flux_cube_nopca, just_tellurics
    return (
        A_noplanet,
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
    Fp_conv,
    Fstar_conv,
    flux_cube,
    n_order,
    n_exposure,
    n_pixel,
    phases,
    Rp_solar,
    Rstar,
    A_noplanet=1,
    do_pca=False,
    n_princ_comp=4,
    star=False,
    observation="emission",
):
    Kstar = 0.3229 * 1.0

    v_sys_measured = (
        1.6845  # this is the systemic velocity of the system reported in the literature
    )

    rv_planet, rv_star = calc_rvs(
        v_sys, v_sys_measured, Kp, Kstar, phases
    )  # measured in m/s

    for order in range(n_order):
        wlgrid_order = np.copy(wlgrid[order,])  # Cropped wavelengths
        model_flux_cube = np.zeros(
            (n_exposure, n_pixel)
        )  # "shifted" model spectra array at each phase
        for exposure in range(n_exposure):
            flux_planet = calc_doppler_shift(
                wlgrid_order, Fp_conv * Rp_solar**2, rv_planet[exposure]
            )
            flux_planet *= scale  # apply scale factor
            flux_star = calc_doppler_shift(
                wlgrid_order, Fstar_conv * Rstar**2, rv_star[exposure]
            )

            if star and observation == "emission":
                model_flux_cube[exposure,] = flux_planet / flux_star + 1.0
            else:  # in transmission, after we "divide out" (with PCA) the star and tellurics, we're left with flux_planet...kinda
                model_flux_cube[exposure,] = 1 - flux_planet

        # ok now do the PCA. where does it fall apart?
        if do_pca:
            # process the model same as the "data"!
            model_flux_cube *= A_noplanet[j]
            model_flux_cube = do_pca(model_flux_cube, n_princ_comp, False)

        # todo: airmass detrending reprocessing
        logL_arr, CCF_arr = calc_ccf_map(model_flux_cube, flux_cube, n_pixel)
        logL = logL_arr.sum()
        CCF = CCF_arr.sum()

    return logL, CCF  # returning CCF and logL values


def run_simulation(
    grid_ind, planet_spectrum_path, star_spectrum_path, phases, observation="emission"
):
    # todo: wrap this in a function? with paths and everything!
    Kp_array = np.linspace(93.06, 292.06, 200)
    v_sys_array = np.arange(-100, 100)
    n_order, n_exposure, n_pixel = (44, 79, 1848)
    scale = 1.0
    mike_wave, mike_cube = np.load(
        "data/data_RAW_20201214_wl_algn_03.pic", allow_pickle=True
    )

    wl_cube_model = mike_wave.copy().astype(np.float64)

    Rvel = np.load(
        "data/rvel.pic", allow_pickle=True
    )  # Time-resolved Earth-star velocity

    # todo: add data in
    # so if I want to change my model, I just alter this!
    wl_model, Fp = np.load(planet_spectrum_path, allow_pickle=True)

    wl_model = wl_model.astype(np.float64)

    Rp = 1.21  # Jupiter radii
    Rp_solar = Rp * rjup_rsun  # convert from jupiter radii to solar radii
    Rstar = 0.955  # solar radii
    Kp = 192.02

    # rotational convolution
    v_rot = 4.5
    Fp_conv_rot = broaden_spectrum(wl_model, Fp, v_rot)

    # instrument profile convolution
    xker = np.arange(41) - 20
    sigma = 5.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # nominal
    yker = np.exp(-0.5 * (xker / sigma) ** 2.0)
    yker /= yker.sum()
    Fp_conv = np.convolve(Fp_conv_rot, yker, mode="same")

    star_wave, star_flux = np.loadtxt(
        star_spectrum_path
    ).T  # Phoenix stellar model packing
    Fstar_conv = get_star_spline(star_wave, star_flux, wl_model, yker, smooth=False)

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

    lls, ccfs = np.zeros((50, 50)), np.zeros((50, 50))

    # redoing the grid. how close does PCA get to a tellurics-free signal detection?
    A_noplanet, flux_cube, flux_cube_nopca, just_tellurics = make_data(
        scale,
        wl_cube_model,
        Fp_conv,
        Fstar_conv,
        n_order,
        n_exposure,
        n_pixel,
        Rp_solar,
        Rstar,
        do_pca=False,
        do_airmass_detrending=True,
        blaze=blaze,
        n_princ_comp=n_princ_comp,
        tellurics=telluric,
        Vsys=0,
        star=star,
        Kp=Kp,
        SNR=SNR,
        tell_type=tell_type,
        time_dep_tell=time_dep_tell,
        wav_error=wav_error,
        order_dep_throughput=order_dep_throughput,
        observation=observation,
    )
    with open(
        f"output/simdata_{n_princ_comp}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt",
        "wb",
    ) as f:
        pickle.dump(flux_cube, f)
    with open(
        f"output/nopca_simdata_{n_princ_comp}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt",
        "wb",
    ) as f:
        pickle.dump(flux_cube_nopca, f)
    with open(
        f"output/A_noplanet_{n_princ_comp}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt",
        "wb",
    ) as f:
        pickle.dump(A_noplanet, f)

    # only save if tellurics are True. Otherwise, this will be cast to an array of ones.
    if tellurics:
        with open(f"output/just_tellurics_vary_airmass.txt", "wb") as f:
            pickle.dump(just_tellurics, f)
    for i, Kp in tqdm(
        enumerate(Kp_array[50:150][::2]), total=50, desc="looping PCA over Kp"
    ):
        for j, v_sys in enumerate(v_sys_array[50:150][::2]):
            res = calc_log_likelihood(
                v_sys,
                Kp,
                scale,
                wl_cube_model,
                Fp_conv,
                Fstar_conv,
                flux_cube,
                n_order,
                n_exposure,
                n_pixel,
                phases,
                Rp_solar,
                Rstar,
                do_pca=True,
                NPC=n_princ_comp,
                A_noplanet=A_noplanet,
                star=star,
            )
            lls[i, j] = res[0]
            ccfs[i, j] = res[1]

    np.savetxt(
        f"output/lls_{n_princ_comp}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt",
        lls,
    )
    np.savetxt(
        f"output/ccfs_{n_princ_comp}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt",
        ccfs,
    )


if __name__ == "__main__":
    ind = eval(sys.argv[1])
    run_simulation(ind)
