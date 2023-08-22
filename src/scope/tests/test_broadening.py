import unittest

from scope.broadening import *


class TestTheta(unittest.TestCase):
    def test_get_theta_zero(self):
        res = get_theta(0, 0)
        self.assertTrue(res == 0)

    def test_get_theta_90_lat(self):
        res = get_theta(np.pi / 2, 0)
        self.assertTrue(res == np.pi / 2)

    def test_get_theta_90_lon(self):
        res = get_theta(0, np.pi / 2)
        self.assertTrue(res == np.pi / 2)

    def test_more_than_each(self):
        """
        at positive lat and lon, there should be more theta than either lat or lon
        """
        res = get_theta(np.pi / 4, np.pi / 4)
        self.assertTrue(res > np.pi / 4)

    def test_get_theta_lat_too_large_positive(self):
        """
        if we get a value that's greater than pi, we should yell.
        """
        self.assertRaises(ValueError, get_theta, 1.1 * np.pi, 0)

    def test_get_theta_lat_too_large_positive(self):
        """
        if we get a value that's greater than pi, we should yell.
        """
        self.assertRaises(
            ValueError,
            get_theta,
            0,
            1.1 * np.pi,
        )


class TestGaussian(unittest.TestCase):
    """
    need to test this! so bad!
    """

    def test_lat_lon_same(self):
        res1 = gaussian_term(0.3, 0, 0, 0.1, 1)
        res2 = gaussian_term(0, 0.3, 0, 0.1, 1)

        self.assertTrue(res1 == res2)

    def test_lat_lon_not_same_with_offset(self):
        res1 = gaussian_term(0.3, 0, 0.3, 0.1, 1)
        res2 = gaussian_term(0, 0.3, 0, 0.1, 1)

        self.assertTrue(res1 > res2)

    def test_broadening_lon(self):
        res1 = gaussian_term(0.3, 0, 0, 0.1, 1)
        res2 = gaussian_term(0.3, 0, 0, 0.2, 1)

        self.assertTrue(res2 > res1)

    def test_broadening_lat(self):
        res1 = gaussian_term(0, 0.3, 0, 0.1, 1)
        res2 = gaussian_term(0, 0.3, 0, 0.2, 1)

        self.assertTrue(res2 > res1)

    def test_broadening_both(self):
        res1 = gaussian_term(0.3, 0.3, 0, 0.1, 1)
        res2 = gaussian_term(0.3, 0.3, 0, 0.2, 1)

        self.assertTrue(res2 > res1)

    def test_amp(self):
        res1 = gaussian_term(0.3, 0.3, 0, 0.1, 1)
        res2 = gaussian_term(0.3, 0.3, 0, 0.2, 2)

        self.assertTrue(res2 > res1)


class TestI_darken(unittest.TestCase):
    """
    test all the functions that do the actual darkening
    """

    def test_I_darken_zero_epsilon(self):
        """
        if there's no epsilon, it's zero at the edge and
        """

        outside = I_darken(np.pi / 4, np.pi / 4, 0)
        center = I_darken(0, 0, 0)
        self.assertTrue(center == outside)

    def test_I_darken_pos_epsilon(self):
        """
        if there's a positive epsilon, it should be less at edge than center.
        """
        outside = I_darken(np.pi / 4, np.pi / 4, 0.3)
        center = I_darken(0, 0, 0.3)
        self.assertTrue(center > outside)

    def test_I_darken_only_pos_epsilon(self):
        """
        should throw an error if epsilon is negative.
        """

        self.assertRaises(ValueError, I_darken, 0, 0, -0.3)

    def test_I_darken_only_epsilon_less_one(self):
        """
        should throw an error if epsilon > 1
        """
        self.assertRaises(ValueError, I_darken, 0, 0, 1.3)

    def test_I_darken_disk_x_greater_one(self):
        """
        any x greater than 1 shouldn't be good.
        """

        self.assertRaises(ValueError, I_darken_disk, 1.5, 0, 0.1)

    def test_I_darken_disk_y_greater_one(self):
        """
        any x greater than 1 shouldn't be good.
        """

        self.assertRaises(ValueError, I_darken_disk, 0, 1.5, 0.1)

    def test_I_darken_disk_same_on_disk(self):
        """
        any x greater than 1 shouldn't be good.
        """
        res1 = I_darken_disk(0, 0.5, 0)
        res2 = I_darken_disk(0.3, 0.1, 0)

        self.assertTrue(res1 == res2)

    def test_I_darken_disk_disk_decrease(self):
        """
        any x greater than 1 shouldn't be good.
        """
        res1 = I_darken_disk(0, 0, 0.3)
        res2 = I_darken_disk(0.3, 0.1, 0.3)

        self.assertTrue(res1 > res2)

    def test_I_darken_disk_disk_decrease(self):
        """
        any x greater than 1 shouldn't be good.
        """
        res1 = I_darken_disk(0, 0, 0.3)
        res2 = I_darken_disk(0.3, 0.1, 0.3)

        self.assertTrue(res1 > res2)

        self.assertTrue(res2 < res1)

        # I_darken_hotspot_add(lon, lat, offset, sigma, epsilon, amp):


class Test_Profile(unittest.TestCase):
    def test_profile_offset_disk(self):
        """
        when it's offset, it should be more redshifted
        """

        self.assertTrue(True)
