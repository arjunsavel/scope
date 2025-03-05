"""
This module contains functions to calculate derived quantities from the input data.
"""
import os
from math import floor

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from exoplanet.orbits.keplerian import KeplerianOrbit

from scope.logger import get_logger
from scope.scrape_igrins_etc import scrape_crires_plus_etc, scrape_igrins_etc

logger = get_logger()

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

    # calculate max npix per resolution element
    calc_max_npix(
        "n_exposures",
        [
            "phase_start",
            "phase_end",
            "P_rot",
            "Mstar",
            "e",
            "Mp",
            "Rstar",
            "omega",
            "b",
            "instrument",
        ],
        data,
    )
    calc_snr("SNR", ["Kmag", "n_exposures", "P_rot", "phase_start", "phase_end"], data)

    return data


def calc_snr(param, args, data):
    detector_reset_times = {
        "IGRINS": 60,
        "CRIRES+": 0,
    }  # these are in seconds. crires+ considers this as part of DIT in the ETC.
    if data[param] == -1:  # only calculate if set to -1
        kmag = data["Kmag"]
        n_exposures = data["n_exposures"]
        instrument = data["instrument"]
        detector_reset_time = detector_reset_times[instrument]
        duration = (
            data["P_rot"] * (data["phase_end"] - data["phase_start"]) * 24 * 3600
        )  # in seconds
        exptime = duration / n_exposures - detector_reset_time
        # ok this is pretty good lol

        if instrument == "IGRINS":
            snr = scrape_igrins_etc(kmag, exptime)
        elif instrument == "CRIRES+":
            snr = scrape_crires_plus_etc(kmag, exptime)
        else:
            logger.error(f"Unknown instrument {instrument}!")
        # need to construct a JSON object to send to the CRRIES+ SNR calculator.
        # ok need to submit the job automatically to the IGRINS SNR calculator.
        data[param] = snr
        data["snr_path"] = os.getcwd() + "/snr.json"


def calc_max_npix(param, args, data):
    # todo: refactor the instrument data to be somwhere!
    instrument_resolutions_dict = {
        "IGRINS": 45000,
        "CRIRES+": 145000,
    }
    pixel_per_res = {"IGRINS": 3.3, "CRIRES+": 3.3}

    if data[param] == -1:
        check_args(args, data, param)
        period = data["P_rot"]
        mstar = data["Mstar"]
        e = data["e"]  # need
        mplanet = data["Mp"]
        rstar = data["Rstar"]
        peri = np.radians(data["omega"])
        b = data["b"]  # need
        R = instrument_resolutions_dict[data["instrument"]]
        pixel_per_res = pixel_per_res[data["instrument"]]
        plot = False
        phase_start = data["phase_start"]
        phase_end = data["phase_end"]
        max_time, max_time_per_pixel, dphase_per_exp, n_exp = calc_crossing_time(
            period=period,
            mstar=mstar,
            e=e,
            mplanet=mplanet,
            rstar=rstar,
            peri=peri,
            b=b,
            R=R,
            pix_per_res=pixel_per_res,
            phase_start=phase_start,
            phase_end=phase_end,
            plot=plot,
        )
        data[param] = floor(n_exp.value)


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


def calc_crossing_time(
    period=1.80988198,
    mstar=1.458,
    e=0.000,
    mplanet=0.894,
    rstar=1.756,
    peri=0,
    b=0.027,
    R=45000,
    pix_per_res=3.3,
    phase_start=0.9668567402328337,
    phase_end=1.0331432597671664,
    plot=False,
):
    """

    todo: refactor this into separate functions, maybe?

    Inputs
    -------
    autofilled for WASP-76b and IGRINS.

    R: (int) resolution of spectrograph.
    pix_per_res: (float) pixels per resolution element


    Outputs
    -------

    min_time: minimum time during transit before lines start smearing across resolution elements.

    min_time_per_pixel: minimum time during transit before lines start smearing across a single pixel.
    dphase_per_exp: the amount of phase (values from 0 to 1) taken up by a single exposure, given above constraints.
    n_exp: number of exposures you can take during transit.

    """
    print(mstar, rstar, period, e, b, peri, mplanet)
    orbit = KeplerianOrbit(
        m_star=mstar,  # solar masses!
        r_star=rstar,  # solar radii!
        #     m_planet_units = u.jupiterMass,
        t0=0,  # reference transit at 0!
        period=period,
        ecc=e,
        b=b,
        omega=np.radians(peri),  # periastron, radians
        Omega=np.radians(peri),  # periastron, radians
        m_planet=mplanet * 0.0009543,
    )
    t = np.linspace(0, period * 1.1, 1000)  # days
    z_vel = orbit.get_relative_velocity(t)[2].eval()

    z_vel *= 695700 / 86400  # km / s

    phases = t / period
    if plot:
        plt.plot(phases, z_vel)

        plt.axvline(0.5, color="gray", label="Eclipse")
        plt.axvline(0.25, label="Quadrature (should be maximized)", color="teal")

        plt.axvline(0.75, label="Quadrature (should be maximized)", color="teal")

        plt.legend()
        plt.xlabel("Time (days)")
        plt.ylabel("Radial velocity (km/s)")
    acceleration = (
        np.diff(z_vel) / np.diff((t * u.day).to(u.s).si.value) * u.km / (u.s**2)
    )

    if plot:
        plt.plot(phases[:-1], acceleration)

        plt.axvline(0.5, color="gray", label="Eclipse")
        plt.axvline(0.25, label="Quadrature (should be minimized)", color="teal")

        plt.axvline(0.75, label="Quadrature (should be minimized)", color="teal")
        plt.xlabel("Orbital phase")
        plt.ylabel("Radial acceleration (km/s^2)")

        # cool, this is the acceleration.

        # now want the pixel crossing time.

        plt.legend()

    # R = c / delta v
    # delta v = c / R

    delta_v = (const.c / R).to(u.km / u.s)
    res_crossing_time = abs(delta_v / acceleration).to(u.s)
    if plot:
        plt.figure()
        plt.plot(phases[:-1], res_crossing_time)
        plt.axvline(0.5, color="gray", label="Eclipse")
        plt.axvline(0.25, label="Quadrature (should be maximized)", color="teal")

        plt.axvline(0.75, label="Quadrature (should be maximized)", color="teal")

        plt.legend()
        plt.xlabel("Orbital phase")
        plt.ylabel("Resolution crossing time (s)")
        plt.yscale("log")

        plt.figure()
        plt.plot(phases[:-1], res_crossing_time)
        plt.legend()

        plt.ylim(820, 900)
        plt.xlabel("Orbital phase")
        plt.ylabel("Resolution crossing time (s)")

    within_phases = (phases[:-1] > phase_start) & (phases[:-1] < phase_end)

    res_crossing_time_phases = res_crossing_time[within_phases]

    max_time = np.min(res_crossing_time_phases)

    max_time_per_pixel = max_time / pix_per_res
    period = period * u.day
    dphase_per_exp = (np.min(res_crossing_time_phases) / period).si

    phasedur = phase_end - phase_start
    n_exp = phasedur / dphase_per_exp
    # todo: actually get phase_start and phase_end in there

    # then query https://igrins-jj.firebaseapp.com/etc/simple:
    # this gives us, for the given exposure time, what the SNR is going to be.
    # well that's more the maximum time
    # so that's the maximum time, but we want more than that many exposures.
    # don't have to worry about pixel-crossing time.
    return max_time, max_time_per_pixel, dphase_per_exp, n_exp
