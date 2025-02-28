"""
tests the run_simulation module.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

from scope.run_simulation import *
from scope.utils import *
from scope.tests.conftest import test_inputs, test_baseline_outouts

# Define the path to the test data
test_data_path = os.path.join(os.path.dirname(__file__), "../data")


# make a pytest fixture to store the test data


@pytest.fixture
def test_noblaze_outputs(test_inputs):
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
        blaze=False,
        tellurics=True,
        n_princ_comp=4,
        v_sys=0,
        Kp=192.06,
        star=True,
        SNR=0,
        rv_semiamp_orbit=0.3229,
        observation="emission",
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


@pytest.fixture
def test_notell_outputs(test_inputs):
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
        tellurics=False,
        n_princ_comp=4,
        v_sys=0,
        Kp=192.06,
        star=True,
        SNR=0,
        rv_semiamp_orbit=0.3229,
        observation="emission",
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


@pytest.fixture
def test_noisy_outputs(test_inputs):
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
        SNR=250,
        rv_semiamp_orbit=0.3229,
        observation="emission",
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


@pytest.fixture
def test_baseline_outputs_take2(test_inputs):
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


def test_make_data_shape(test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts

    # assert that the shape is as expected
    assert flux_cube.shape == (n_order, n_exposure, n_pixel)


def test_make_data_not_all_zeros(test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts

    # assert that the shape is as expected
    assert not np.all(flux_cube == 0.0)


# def test_tellurics_cube_normalized_correctly(test_baseline_outouts):
#     """
#     Tests the make_data function.
#     """
#
#     A_noplanet, flux_cube, flux_cube_nopca, just_tellurics, n_exposure, n_order, n_pixel = test_baseline_outouts
#
#     # assert that the shape is as expected
#     assert np.max(just_tellurics) == 1.0


def test_flux_values_reasonable(test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts

    # assert that the shape is as expected
    assert np.median(flux_cube_nopca) > 0.1


def test_tellurics_regular_values(test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts

    # assert that the shape is as expected
    assert np.median(just_tellurics) > 0.1


def test_tellurics_actually_are_penalized(test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts

    # assert that the shape is as expected
    assert np.median(just_tellurics) < 1.0


def test_all_finite_made_data(test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts

    # assert that the shape is as expected
    assert np.all(np.isfinite(flux_cube))


def test_nopca_fluxlevels_blaze(test_noblaze_outputs, test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_noblaze_outputs
    (
        A_noplanet_b,
        flux_cube_b,
        flux_cube_nopca_b,
        just_tellurics_b,
        n_exposure_b,
        n_order_b,
        n_pixel_b,
    ) = test_baseline_outouts

    # the flux levels must be higher when there's no blaze.
    assert np.median(flux_cube_nopca) > np.median(flux_cube_nopca_b)


def test_nopca_fluxlevels_tell(test_notell_outputs, test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_notell_outputs
    (
        A_noplanet_b,
        flux_cube_b,
        flux_cube_nopca_b,
        just_tellurics_b,
        n_exposure_b,
        n_order_b,
        n_pixel_b,
    ) = test_baseline_outouts

    # the flux levels must be higher when there's no tellurics.
    assert np.median(flux_cube_nopca) > np.median(flux_cube_nopca_b)


def test_nopca_fluxlevels_tell(test_notell_outputs, test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_notell_outputs
    (
        A_noplanet_b,
        flux_cube_b,
        flux_cube_nopca_b,
        just_tellurics_b,
        n_exposure_b,
        n_order_b,
        n_pixel_b,
    ) = test_baseline_outouts

    # the flux levels must be higher when there's no tellurics.
    assert np.median(flux_cube_nopca) > np.median(flux_cube_nopca_b)


def test_nopca_fluxlevels_noisier(test_noisy_outputs, test_baseline_outouts):
    """
    Tests the make_data function.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_noisy_outputs
    (
        A_noplanet_b,
        flux_cube_b,
        flux_cube_nopca_b,
        just_tellurics_b,
        n_exposure_b,
        n_order_b,
        n_pixel_b,
    ) = test_baseline_outouts

    # the flux levels must be higher when there's no tellurics.
    assert np.std(flux_cube) > np.std(flux_cube_b)


def test_nopca_fluxlevels_deterministic(
    test_baseline_outputs_take2, test_baseline_outouts
):
    """
    without any noise specified, everything should be deterministic.
    """

    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outputs_take2
    (
        A_noplanet_b,
        flux_cube_b,
        flux_cube_nopca_b,
        just_tellurics_b,
        n_exposure_b,
        n_order_b,
        n_pixel_b,
    ) = test_baseline_outouts

    # the flux levels must be higher when there's no tellurics.
    np.testing.assert_array_equal(flux_cube_nopca, flux_cube_nopca_b)


def test_crires_simulation(test_inputs):
    data_cube_path = os.path.join(test_data_path, "data_RAW_20201214_wl_algn_03.pic")
    planet_spectrum_path = os.path.join(
        test_data_path, "best_fit_spectrum_tran_gj1214_steam_nir.pic"
    )
    snr_path = os.path.join(test_data_path, "output1.json")
    star_spectrum_path = os.path.join(test_data_path, "PHOENIX_5605_4.33.txt")
    simulate_observation(
        planet_spectrum_path=planet_spectrum_path,
        star_spectrum_path=star_spectrum_path,
        data_cube_path=data_cube_path,
        phase_start=0,
        phase_end=1,
        n_exposures=2,
        observation="emission",
        blaze=True,
        n_princ_comp=4,
        star=True,
        SNR=250,
        telluric=False,
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
        planet_name="GJ1214b",
        divide_out_of_transit=False,
        out_of_transit_dur=0.1,
        include_rm=False,
        v_rot_star=3.0,
        a=0.033,  #
        lambda_misalign=0.0,
        inc=90.0,
        seed=42,
        vary_throughput=True,
        snr_path=snr_path,
    )

    # check that there are nonzeros, basically that it runs
    filetype = glob("output/yourfirstsimulation/*ccf*")
    ccfs = np.loadtxt(filetype[0])
    assert np.sum(ccfs) > 0
