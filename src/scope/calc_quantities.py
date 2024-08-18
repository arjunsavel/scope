import astropy.units as u
import numpy as np

# refactor the below two functions to be a single function


def calc_velocity(distance, P, distance_unit=u.AU, time_unit=u.day):
    """
    Calculate a velocity over some period over some distance.
    """
    # Calculate the velocity
    velocity = 2 * np.pi * distance * distance_unit / (P * time_unit)

    return velocity.to(u.km / u.s).value
