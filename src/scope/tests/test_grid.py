"""
goals: make sure that grid can be indexed, no duplicates, etc.
"""

import unittest

from scope.grid import *


class TestGrid(unittest.TestCase):
    def test_indexing(self):
        dictionary = parameter_list[0]
        self.assertTrue(type(dictionary) == dict)

    def test_no_duplicates(self):
        new_list = []
        for dictionary in parameter_list:
            if dictionary not in new_list:
                new_list += [dictionary]
        self.assertTrue(len(new_list) == len(parameter_list))
