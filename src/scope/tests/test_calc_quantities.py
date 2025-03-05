import unittest
from scope.calc_quantities import calc_crossing_time


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
