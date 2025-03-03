import os
import sys
import pytest
import numpy as np
import pandas as pd

from scope.run_simulation import *
from scope.utils import *

# Define the path to the test data
test_data_path = os.path.join(os.path.dirname(__file__), "../data")


@pytest.fixture
def test_inputs():
    """
    sets up the spectral things.
    """
    data_cube_path = os.path.join(test_data_path, "data_RAW_20201214_wl_algn_03.pic")
    planet_spectrum_path = os.path.join(
        test_data_path, "best_fit_spectrum_tran_gj1214_steam_nir.pic"
    )
    wl_model, Fp, Fstar = np.load(planet_spectrum_path, allow_pickle=True)
    mike_wave, mike_cube = pickle.load(open(data_cube_path, "rb"), encoding="latin1")
    wl_cube_model = mike_wave.copy().astype(np.float64)

    v_rot = 4.33  # km/s
    rot_ker = get_rot_ker(v_rot, wl_model)
    Fp_conv_rot = np.convolve(Fp, rot_ker, mode="same")
    # instrument profile convolution
    instrument_kernel = get_instrument_kernel("IGRINS")
    Fp_conv = np.convolve(Fp_conv_rot, instrument_kernel, mode="same")
    star_spectrum_path = os.path.join(test_data_path, "PHOENIX_5605_4.33.txt")
    star_wave, star_flux = np.loadtxt(
        star_spectrum_path
    ).T  # Phoenix stellar model packing
    Fstar_conv = get_star_spline(
        star_wave, star_flux, wl_model, instrument_kernel, smooth=False
    )

    return Fp_conv, Fstar_conv, wl_cube_model, wl_model


@pytest.fixture
def test_baseline_outouts(test_inputs):
    Fp_conv, Fstar_conv, wl_cube_model, wl_model = test_inputs

    scale = 1

    n_exposure = 10
    n_order, n_pixel = (44, 1848)  # todo: fix.
    phases = np.linspace(-0.01, 0.01, n_exposure)
    Rp_solar = 0.1
    Rstar = 1.0

    # Test the function
    A_noplanet, flux_cube, flux_cube_nopca, just_tellurics = make_data(
        scale,
        wl_cube_model,
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
        blaze=True,
        tellurics=True,
        n_princ_comp=4,
        v_sys=0,
        Kp=192.06,
        star=True,
        SNR=0,
        rv_semiamp_orbit=0.3229,
        observation="transmission",
        tell_type="data-driven",
        time_dep_tell=False,
        wav_error=False,
        order_dep_throughput=True,
        a=1,  # AU
        u1=0.3,
        u2=0.3,
        LD=True,
        b=0.0,  # impact parameter
        divide_out_of_transit=False,
        out_of_transit_dur=0.1,
    )
    return (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    )
