import astropy.units as u
import numpy as np


def calc_v_rot(Rp, P):
    """
    Calculate the rotational velocity of a planet.

    Parameters
    ----------
    Rp : float
        The radius of the planet in Jupiter radii.
    P : float
        The period of the planet in days.

    Returns
    -------
    v_rot : float
        The rotational velocity of the planet in km/s.
    """
    # Calculate the rotational velocity
    v_rot = 2 * np.pi * Rp * u.R_jup / (P * u.day)

    return v_rot.to(u.km / u.s).value


def calc_kp(a, P):
    """
    Calculate the orbital velocity of the planet.
    """
    # Calculate the orbital velocity
    kp = 2 * np.pi * a * u.AU / (P * u.day)

    return kp.to(u.km / u.s).value
