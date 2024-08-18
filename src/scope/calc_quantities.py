import astropy.units as u
import numpy as np

# refactor the below two functions to be a single function


def check_args(args, data, quantity):
    """
    Check that the arguments are in the data dictionary.

    Parameters
    ----------
    args : list
        List of arguments to check.
    data : dict
        Dictionary of data.

    """
    for arg in args:
        if arg not in data:
            raise ValueError(
                f"{arg} not in data dictionary! It is required to calculate {quantity}!"
            )


def calculate_derived_parameters(data):
    """
    Calculate derived parameters from the input data.

    """
    # first, equatorial rotational velocity.

    def calc_param_boilerplate(param, args, data, distance_unit):
        if np.isnan(data[param]):
            check_args(args, data, param)
            data[param] = calculate_velocity(
                *[data[arg] for arg in args], distance_unit=distance_unit
            )

    # Calculate the equatorial rotational velocity
    calc_param_boilerplate("v_rot", ["Rp", "P_rot"], data, u.R_jup)

    # calculate planetary orbital velocity
    calc_param_boilerplate("kp", ["a", "P_rot"], data, u.AU)

    return data


def calculate_velocity(distance, P, distance_unit=u.AU, time_unit=u.day):
    """
    Calculate a velocity over some period over some distance.
    """
    # Calculate the velocity
    velocity = 2 * np.pi * distance * distance_unit / (P * time_unit)

    return velocity.to(u.km / u.s).value


def convert_tdur_to_phase(tdur, period):
    """
    Convert the transit duration to a phase.

    Parameters
    ----------
    tdur : float
        The transit duration in hours.
    period : float
        The period of the planet in days.

    Returns
    -------
    float
        The phase of the transit duration.
    """
    return ((tdur * u.hour) / (period * u.day)).si.value
