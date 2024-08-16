"""
Add noise to simulated data.
"""
import numpy as np

from scope.utils import *


def add_constant_noise(flux_cube_model, wl_grid, SNR):
    """
    Adds constant noise to the flux cube model.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio

    Outputs
    -------
        :noisy_flux: (array) flux cube model with constant noise added.
    """
    n_photons = SNR**2
    # SNR**2 is the total number of photons.
    flux_cube = (
        flux_cube_model.copy() * n_photons
    )  # now scaled by number of photons. it's the maximum

    noise_matrix = np.random.normal(loc=0, scale=1, size=flux_cube_model.shape)
    noise_matrix_scaled = noise_matrix * np.sqrt(flux_cube)
    noisy_flux = flux_cube + noise_matrix_scaled
    return noisy_flux


def add_quadratic_noise(flux_cube_model, wl_grid, SNR, IGRINS=False, **kwargs):
    """
    Currently assumes that there are two bands: A and B.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio
        :IGRINS: (bool) if True, then the data is IGRINS data. If False, then the data is NOT IGRINS data.

    Outputs
    -------
        :noisy_flux: (array) flux cube model with quadratic noise added.
    """
    if IGRINS:
        A_a, A_b, A_c, B_a, B_b, B_c = np.loadtxt("data/igrins_median_snr.txt")
        # this is scaled to a mean SNR of 250.
        noisy_flux = np.ones_like(flux_cube_model) * 0
        wl_cutoff = 1.9
        medians = np.median(wl_grid, axis=1)

        A_wl = medians[medians < wl_cutoff]
        B_wl = medians[medians > wl_cutoff]

        A_SNRs = A_a + A_b * A_wl + A_c * A_wl**2
        B_SNRs = B_a + B_b * B_wl + B_c * B_wl**2

        # need to scale these to the required SNR.
        A_SNRs *= SNR / 250
        B_SNRs *= SNR / 250
        noisy_flux = np.ones_like(flux_cube_model) * 0

        # iterate through each exposure
        for exposure in range(flux_cube_model.shape[1]):
            for order in range(flux_cube_model.shape[0]):
                if medians[order] < wl_cutoff:  # first band
                    order_snr = A_SNRs[order - len(B_SNRs)]
                else:  # second band
                    order_snr = B_SNRs[order]
                flux_level = (
                    (order_snr**2)
                    * flux_cube_model[order][exposure]
                    / np.nanmax(flux_cube_model[order][exposure])
                )
                noisy_flux[order][exposure] = np.random.poisson(flux_level)

                # a few quick checks to make sure that nothing has gone wrong with adding noise
                noisy_flux[order][exposure][noisy_flux[order][exposure] < 0.0] = 0.0
                noisy_flux[order][exposure][
                    ~np.isfinite(noisy_flux[order][exposure])
                ] = 0.0

    else:
        raise NotImplementedError("Only IGRINS data is currently supported.")

    return noisy_flux


def add_igrins_noise(flux_cube_model, wl_grid, SNR):
    """
    Adds IGRINS noise to the flux cube model.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio

    Outputs
    -------
        :noisy_flux: (array) flux cube model with IGRINS noise added.
    """
    return add_quadratic_noise(flux_cube_model, wl_grid, SNR, IGRINS=True)


def add_custom_noise(SNR):
    """
    Adds custom noise to the flux cube model.
    """
    raise NotImplementedError("Custom noise is not yet implemented.")


def add_noise_cube(flux_cube_model, wl_grid, SNR, noise_model="constant", **kwargs):
    """
    Per the equation in Brogi + Line 19.
    Assumes that the flux cube is scaled 0 to 1.

    I guess note that when I say "SNR", I'm ignoring for now the wavelength-dependent
    flux that you get because of a blaze function. It's the SNR at the peak of the blaze function.

    Inputs
    ------
        :flux_cube_model: (array) flux cube model
        :wl_grid: (array) wavelength grid
        :SNR: (float) signal-to-noise ratio
        :noise_model: (str) noise model to use. Can be constant, IGRINS, custom_quadratic, or custom.

    Outputs
    -------
        :noisy_flux: (array) flux cube model with noise added.
    """

    noise_models = {
        "constant": add_constant_noise,
        "IGRINS": add_igrins_noise,
        "custom_quadratic": add_quadratic_noise,
        "custom": add_custom_noise,
    }
    if noise_model not in noise_models.keys():
        raise ValueError(
            "Noise model can only be constant, IGRINS, custom_quadratic, or custom."
        )

    noise_func = noise_models[noise_model]

    noisy_flux = noise_func(flux_cube_model, wl_grid, SNR, **kwargs)

    # need to make sure that it's still normed 0 to 1!
    noisy_flux = detrend_cube(noisy_flux, noisy_flux.shape[0], noisy_flux.shape[1])

    return noisy_flux
