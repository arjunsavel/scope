import unittest

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
        detrended_cube = detrend_cube(self.test_cube, n_order, n_exp)
        self.assertTrue(orig_min_arr == np.argmin(detrended_cube, axis=0))

    def test_cube_normalized_max_still_max(self):
        orig_min_arr = np.argmax(self.test_cube, axis=0)
        detrended_cube = detrend_cube(self.test_cube, n_order, n_exp)
        self.assertTrue(orig_min_arr == np.argmax(detrended_cube, axis=0))
