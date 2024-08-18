import astropy.units as u
import numpy as np

# refactor the below two functions to be a single function


def calculate_derived_parameters(data):
    """
    Calculate derived parameters from the input data.

    """
    # first, equatorial rotational velocity.
    if np.isnan(data["v_rot"]):
        if np.isnan(data["Rp"]) or np.isnan(data["P_rot"]):
            raise ValueError("Rp and P must be provided to calculate v_rot!")
        data["v_rot"] = calculate_velocity(
            data["Rp"], data["P_rot"], distance_unit=u.R_jup
        )

    # second, orbital velocity of the planet.
    if np.isnan(data["kp"]):
        if np.isnan(data["a"]) or np.isnan(data["P_rot"]):
            raise ValueError("a and P must be provided to calculate kp!")
        data["kp"] = calculate_velocity(data["a"], data["P_rot"], distance_unit=u.AU)

    return data


def calculate_velocity(distance, P, distance_unit=u.AU, time_unit=u.day):
    """
    Calculate a velocity over some period over some distance.
    """
    # Calculate the velocity
    velocity = 2 * np.pi * distance * distance_unit / (P * time_unit)

    return velocity.to(u.km / u.s).value
