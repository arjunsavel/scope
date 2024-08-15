"""
This file tests the cross-correlation function.
"""

import numpy as np
import pytest

from scope.ccf import calc_ccf


def test_autocorr_peaks_zero():
    """
    Tests that the autocorrelation peaks at zero.
    """
    n_pixel = 101
    x = np.arange(n_pixel)
    y = np.sin(x)
    model_flux = np.array([y, y])
    # normalize arrays correctly. subtract mean and divide standard deviation
    model_flux = (model_flux - np.mean(model_flux, axis=1)[:, None]) / np.std(
        model_flux, axis=1
    )[:, None]

    # now shift the data in a loop to find the maximum CCF

    model_flux_slice = model_flux.copy()

    CCF_arr = np.ones(n_pixel)
    for i in range(0, n_pixel):
        model_flux_slice = np.roll(model_flux, i, axis=1)
        logl, CCF = calc_ccf(model_flux, model_flux_slice, n_pixel)
        CCF_arr[i] = CCF

    assert np.argmax(CCF_arr) == 0
