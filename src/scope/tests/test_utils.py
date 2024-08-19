import unittest

import numpy as np

import pytest

from scope.utils import *


class Testdetrend_cube(unittest.TestCase):
    test_cube = np.random.random((10, 10, 10))

    def test_cube_normalized(self):
        n_order, n_exp, n_pix = self.test_cube.shape
        detrended_cube = detrend_cube(self.test_cube, n_order, n_exp)
        self.assertTrue(np.max(detrended_cube) == 1)

    def test_cube_normalized_same_shape(self):
        original_shape = self.test_cube.shape
        n_order, n_exp, n_pix = original_shape
        detrended_cube = detrend_cube(self.test_cube, n_order, n_exp)
        self.assertTrue(detrended_cube.shape == original_shape)

    def test_cube_normalized_min_still_min(self):
        orig_min_arr = np.argmin(self.test_cube, axis=0)
        n_order, n_exp, n_pix = self.test_cube.shape
        detrended_cube = detrend_cube(self.test_cube, n_order, n_exp)
        self.assertTrue(np.all(orig_min_arr == np.argmin(detrended_cube, axis=0)))

    def test_cube_normalized_max_still_max(self):
        orig_min_arr = np.argmax(self.test_cube, axis=0)
        n_order, n_exp, n_pix = self.test_cube.shape
        detrended_cube = detrend_cube(self.test_cube, n_order, n_exp)
        self.assertTrue(np.all(orig_min_arr == np.argmax(detrended_cube, axis=0)))


# todo: benchmark against batman at some point.


@pytest.mark.parametrize(
    "values, output",
    [
        ([0, 0, 1, 0, 1, 0, True], 1.0),  # if no coefficients, no LD
        ([0, 0, 1, 0, 1, 0, False], 1.0),  # but that doesn't matter if LD is turned off
        (
            [1, 1, 10 * rsun, 0, 1e-3, 0.3, True],
            0.0,
        ),  # if there is a really tiny sun, you're not gonna hit the sun
        ([1, 1, 1, 0, 1, 0, True], 1.0),  # at center phase, should be 1
        (
            [0.3, 0.3, 10 * rsun, 0.0, 1, 0.25, True],
            0.0,
        ),  # at quadrature you're definitely not transiting!
        (
            [0.3, 0.3, 10 * rsun, 1.3, 1, 0.0, True],
            0.0,
        ),  # non-transiting planets don't transit!
    ],
)
def test_calc_limb_darkening(values, output):
    assert calc_limb_darkening(*values) == output


class TestPca(unittest.TestCase):
    """
    These don't really test the fancy statistics of PCA. just assert that it's a
    variance-reducing black box.
    """

    n_exp = 10
    n_pix = 10
    cube = np.random.random((n_exp, n_pix))

    def test_perform_pca_shapes(self):
        """
        just make sure the output shapes are what we expect.
        """

        n_princ_comp = 5
        scaled_cube, pca = perform_pca(self.cube, n_princ_comp, return_noplanet=True)
        assert scaled_cube.shape == self.cube.shape
        assert pca.shape == self.cube.shape

    def test_perform_pca_variance_removed(self):
        """
        just make sure the output shapes are what we expect.
        """

        n_princ_comp = 5
        scaled_cube, pca = perform_pca(self.cube, n_princ_comp, return_noplanet=True)
        assert np.std(scaled_cube) < np.std(
            pca
        )  # the variance should be contained in the main components

    def test_perform_pca_variance_removed_compared_input(self):
        """
        just make sure the output shapes are what we expect.
        """

        n_princ_comp = 5
        scaled_cube, pca = perform_pca(self.cube, n_princ_comp, return_noplanet=True)
        assert np.std(scaled_cube) < np.std(
            self.cube
        )  # the variance should be contained in the main components

    def test_perform_pca_variance_removed_compared_input_diffcompoents(self):
        """
        just make sure the output shapes are what we expect.
        """

        n_princ_comp = 5
        scaled_cube5, pca5 = perform_pca(self.cube, n_princ_comp, return_noplanet=True)
        n_princ_comp = 10
        scaled_cube10, pca10 = perform_pca(
            self.cube, n_princ_comp, return_noplanet=True
        )
        assert np.std(scaled_cube10) < np.std(
            scaled_cube5
        )  # the variance should be contained in the main components


class TestDopplerShift(unittest.TestCase):
    def test_same_after_shift(self):
        """
        shouldn't globally change the properties. should really just shift things.
        """
        eval_wave = np.linspace(1.1, 1.8, 10000) * 1e-6  # meters.
        template_wave = np.linspace(1, 2, 10000) * 1e-6

        template_flux = np.random.random(10000)
        interped_flux = np.interp(eval_wave, template_wave, template_flux)
        v = 1e-6  # m/s. should not change much

        shifted_flux = calc_doppler_shift(eval_wave, template_wave, template_flux, v)
        # test that shifted flux and interpred flux are very similar
        assert (
            np.testing.assert_allclose(shifted_flux, interped_flux, rtol=1e-2) == None
        )

    def test_diff_after_shift(self):
        """
        shouldn't globally change the properties. should really just shift things.
        """
        eval_wave = np.linspace(1.1, 1.8, 10000) * 1e-6  # meters.
        template_wave = np.linspace(1, 2, 10000) * 1e-6

        template_flux = np.random.random(10000)
        interped_flux = np.interp(eval_wave, template_wave, template_flux)
        v = 2e3  # m/s. should not change much

        shifted_flux = calc_doppler_shift(eval_wave, template_wave, template_flux, v)
        # test that shifted flux and interpred flux are very similar
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, shifted_flux, interped_flux
        )

    def test_gaussian_shifted_blue(self):
        """
        shouldn't globally change the properties. should really just shift things.
        """
        eval_wave = np.linspace(1.1, 1.8, 10000) * 1e-6
        template_wave = np.linspace(1, 2, 10000) * 1e-6
        # make a gaussian centered at 1.5
        template_flux = np.exp(-0.5 * ((template_wave - 1.5e-6) / 0.01) ** 2)
        interped_flux = np.interp(eval_wave, template_wave, template_flux)

        v = -5e4  # m/s. should not change much
        shifted_flux = calc_doppler_shift(eval_wave, template_wave, template_flux, v)
        wav_max_shifted = eval_wave[np.argmax(shifted_flux)]
        wav_max = eval_wave[np.argmax(interped_flux)]
        assert wav_max_shifted < wav_max

    def test_gaussian_shifted_red(self):
        """
        shouldn't globally change the properties. should really just shift things.
        """
        eval_wave = np.linspace(1.1, 1.8, 10000) * 1e-6
        template_wave = np.linspace(1, 2, 10000) * 1e-6
        # make a gaussian centered at 1.5
        template_flux = np.exp(-0.5 * ((template_wave - 1.5e-6) / 0.01) ** 2)

        v = 5e4  # m/s. should not change much
        shifted_flux = calc_doppler_shift(eval_wave, template_wave, template_flux, v)
        interped_flux = np.interp(eval_wave, template_wave, template_flux)
        wav_max_shifted = eval_wave[np.argmax(shifted_flux)]
        wav_max = eval_wave[np.argmax(interped_flux)]
        assert wav_max_shifted > wav_max


class TestCrossingTime(unittest.TestCase):
    """
    tests the exposure time calculation.
    """

    period = 1.80988198
    mstar = 1.458
    e = 0.000
    inc = 89.623
    mplanet = 0.894
    rstar = 1.756
    peri = 0
    b = 0.027
    R = 45000
    pix_per_res = 3.3
    plot = False

    def test_crossing_time_moremassive_star(self):
        """
        in a greater potential well, the crossing time is shorter.
        """
        low_mass_time, _, _, _ = calc_crossing_time()

        high_mass_time, _, _, _ = calc_crossing_time(mstar=10 * self.mstar)
        assert low_mass_time > high_mass_time

    def test_crossing_time_moremassive_star(self):
        """
        on a longer period for the same star, the crossing time is longer.
        """
        short_period_time, _, _, _ = calc_crossing_time()

        long_period_time, _, _, _ = calc_crossing_time(period=10 * self.period)
        assert short_period_time < long_period_time

    def test_crossing_time_pixel_per_res(self):
        """
        more pixels per resolution element, shorter time to cross the pixels.
        """
        _, few_pixels_per_element, _, _ = calc_crossing_time()

        _, many_pixels_per_element, _, _ = calc_crossing_time(
            pix_per_res=10 * self.pix_per_res
        )
        assert many_pixels_per_element < few_pixels_per_element

    def test_crossing_time_resolution(self):
        """
        higher resolution, shorter crossing time.
        """
        low_resolution_time, _, _, _ = calc_crossing_time()

        high_resolution_time, _, _, _ = calc_crossing_time(R=10 * self.R)
        assert high_resolution_time < low_resolution_time
