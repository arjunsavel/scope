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
