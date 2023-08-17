"""
file to calculate the intensity map and broadening said map. and the velocity profile.
"""

import numpy as np
from numba import njit


@njit
def get_theta(lat, lon):
    """
    Converts latitude and longitude to Theta, the angular distance across the disk.

    Inputs
    ------
        :lat: latitude in radians
        :lon: longitude in radians

    Outputs
    -------
        :res1: the angular distance across the disk.

    """
    if (-np.pi <= lon <= np.pi) and (-np.pi <= lat <= np.pi):
        return np.arccos(np.sqrt(1 - np.sin(lon) ** 2 - np.sin(lat) ** 2))
    else:
        raise ValueError(
            "lat or lon is too large. are you inputting in radians? you should be!"
        )


@njit
def I_darken(lon, lat, epsilon):
    """
    takes lon and lat in. spits out the value on the limb-darkened disk.

    Inputs
    ------
        :lon: longitude in radians
        :lat: latitude in radians
        :epsilon: limb-darkening coefficient. should be between 0 and 1.

    Outputs
    -------
        :res1: the intensity at that point on the disk.
    """
    if 0.0 <= epsilon <= 1.0:
        theta = get_theta(lat, lon)
        res1 = (1 - epsilon) + (epsilon * np.cos(theta))
        return res1
    else:
        raise ValueError("epsilon should only be from 0 to 1")


@njit
def I_darken_disk(x, y, epsilon):
    """
    takes x and y in. spits out the value on the limb-darkened disk.

    Inputs
    ------
        :x: x position on the disk
        :y: y position on the disk
        :epsilon: limb-darkening coefficient. should be between 0 and 1.

    Outputs
    -------
        :res1: the intensity at that point on the disk.
    """
    lon = np.arcsin(x)
    lat = np.arcsin(y)

    return I_darken(lon, lat, epsilon)


@njit
def gaussian_term(lon, lat, offset, sigma, amp):
    """
    the gaussian term
    Inputs
    ------
        :lat:
        :lon:
        :offset:
        :sigma:
        :amp:

    Outputs
    -------

    """
    return (
        amp
        * (1 / (sigma**2 * 2 * np.pi))
        * np.exp(-((lon - offset) ** 2) / (2 * sigma**2))
        * np.exp(-((lat) ** 2) / (2 * sigma**2))
    )


@njit
def gaussian_term_1d(lat, offset, sigma, amp):
    """
    the gaussian term. this time there's no longitude dependence!
    Inputs
    ------
        :lat:
        :lon:
        :offset:
        :sigma:
        :amp:

    Outputs
    -------

    """
    return (
        amp
        * (1 / (sigma**2 * 2 * np.pi))
        * np.exp(-((lat - offset) ** 2) / (2 * sigma**2))
    )


#### below: the actual integrations

vl = 1


@njit
def numerator_integral_no_sigma(vz, epsilon, n_samples=100):
    """
    calculates the numerator of the line profile. this is the version without the exponent.

    Parameters
    ----------
    vz : float
        The velocity of the star.
    epsilon : float
        The limb-darkening coefficient.
    n_samples : int
        The number of samples to use in the integration.

    Returns
    -------
    total_res : float
        The numerator of the line profile.
    """
    # set bounds
    lower_bound = 0
    upper_bound = np.arcsin(np.sqrt(1 - np.square(vz / vl)))
    # need to fix these bounds
    lat_arr = np.linspace(lower_bound, upper_bound, n_samples)
    d_lat = np.diff(lat_arr)[0]

    total_res = 0.0
    for lat in lat_arr:
        res = (
            (
                (1 - epsilon)
                + epsilon * np.sqrt(1 - np.square(vz / vl) - np.square(np.sin(lat)))
            )
            * np.cos(lat)
            * d_lat
        )
        if not np.isnan(res):
            total_res += res
    return total_res


@njit
def numerator_no_sigma(vz, epsilon, n_samples=100):
    """
    calculates the numerator of the line profile. this is the version without the exponent.

    Parameters
    ----------
    vz : float
        The velocity of the star.
    epsilon : float
        The limb-darkening coefficient.
    n_samples : int
        The number of samples to use in the integration.

    Returns
    -------
    res1 : float
        The numerator of the line profile.
    """
    res1 = 2
    return res1 * numerator_integral_no_sigma(vz, epsilon, n_samples=n_samples)


@njit
def line_profile_no_exponent(epsilon, dRV, n_samples=int(1e4), model="igrins", vl=5):
    """
    calculates the line profile for a given epsilon and dRV. this is the version without the exponent.

    Parameters
    ----------
    epsilon : float
        The limb-darkening coefficient.
    dRV : float
        The width of the line profile.
    n_samples : int
        The number of samples to use in the integration.
    model : str
    vl : float
        The equatorial rotational velocity of the star.

    Returns
    -------
    big_resoluts : np.ndarray
        The line profile.

    """
    if model == "igrins":
        n_vz = 2 * vl / dRV
    else:
        n_vz = 500
    vzs = np.linspace(
        -1, 1, int(n_vz)
    )  # need to set to 0 if off disk. this is actually vz / vl. can also step off.
    big_resoluts = np.zeros(vzs.shape)

    for i, vz in enumerate(vzs):
        re_num = numerator_no_sigma(vz, epsilon, n_samples=n_samples)
        #         vz, epsilon, mu, sigma=.7, n_samples=100

        # todo: check sensitivity to lat / lon gridding
        re_denom = I_darken_disk(X, Y, epsilon)

        big_resoluts[i] = re_num / re_denom

    # normalize
    big_resoluts /= np.sum(big_resoluts)
    return big_resoluts


def convert_vz_to_lon(vz, R, sigma_jet, amp_jet, mu_jet):
    """
    Converts vz to a longitude.

    Parameters
    ----------
    vz : float
        The velocity along the line of sight.
    R : float
        The radius of the star.
    sigma_jet : float
        The width of the jet.
    amp_jet : float
        The amplitude of the jet.
    mu_jet : float
        The center of the jet.


    """
    gauss_term = gaussian_term_1d(vz, sigma_jet, amp_jet, mu_jet)
    return np.arcsin(vz / (R * gauss_term))


def convert_lon_to_vz(lon, R, sigma_jet, amp_jet, mu_jet):
    """
    Converts longitude to a vz

    Parameters
    ----------
    lon : float
        The longitude.
    R : float
        The radius of the planet.
    sigma_jet : float
        The width of the jet.
    amp_jet : float
        The amplitude of the jet.
    mu_jet : float
        The center of the jet.


    """
    vz = (
        R * np.sin(lon) * gaussian_term_1d(lon, sigma_jet, amp_jet, mu_jet)
    )  # todo: check another factor of R?
    return vz


@njit
def broaden_spectrum(
    wav, spectrum_flux, epsilon, amp, n_samples=int(1e4), vl=5, model="igrins"
):
    """
    Broadens a spectrum by a line profile.

    Parameters
    ----------
    wl_model : array-like
        The wavelength model.
    spectrum_flux : array-like
        The flux of the spectrum.
    epsilon : float
        The spot contrast.
    n_samples : int
        The number of samples to use in the integration.
    vl : float
        The rotational velocity of the object.

    Returns
    -------
    array-like
        The broadened spectrum.
    """
    dRV = (
        np.mean(np.concatenate([np.diff(wav) / wav[1:], np.diff(wav) / wav[:-1]]))
        * const_c
        * 1e3
    )
    profile = line_profile_no_exponent(
        epsilon, dRV, n_samples=n_samples, vl=vl, model=model
    )
    return np.convolve(spectrum_flux, profile, mode="same")


x = np.linspace(-1, 1, 80)
y = np.linspace(-1, 1, 80)

X, Y = np.meshgrid(x, y)

dx = np.diff(x)[0]
dy = np.diff(y)[0]

if __name__ == "__main__":
    x = np.linspace(-1, 1, 80)
    y = np.linspace(-1, 1, 80)

    vl = 5

    X, Y = np.meshgrid(x, y)

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    lon = np.linspace(-90, 90, 80)
    lat = np.linspace(-90, 90, 80)

    lon = np.radians(lon)
    lat = np.radians(lat)

    LON, LAT = np.meshgrid(lon, lat)

    x = np.linspace(-1, 1, 80)
    y = np.linspace(-1, 1, 80)

    X, Y = np.meshgrid(x, y)

    res = I_darken(LON, LAT, 0.1)
    res_disk = I_darken_disk(X, Y, 0.9)
