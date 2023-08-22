import unittest

from scope.constants import *


class TestC(unittest.TestCase):
    def test_c_large(self):
        self.assertTrue(const_c > 3e8)

    def test_c_small(self):
        self.assertTrue(const_c < 4e8)


class TestRjup(unittest.TestCase):
    def test_rjup_small(self):
        self.assertTrue(rjup_rsun < 0.2)

    def test_rjup_large(self):
        self.assertTrue(rjup_rsun > 0.1)
