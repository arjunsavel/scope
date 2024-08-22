"""
Read in and apply tellurics to simulated data.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from scipy import interpolate

abs_path = os.path.dirname(__file__)


def read_atran(path):
    """
    reads a single ATRAN file.

    Inputs
    ------
    path: string
        Path to the file.

    Returns
    -------
    atran: dataframe
        Atran file.
    """
    atran = pd.read_csv(
        path,
        index_col=0,
        names=["wav", "depth"],
        sep="\s+",
    ).drop_duplicates(subset=["wav"])

    return atran


def read_atran_pair(path1, path2):
    """
    reads and concatenates an ATRAN file pair.
    """
    atran_40_first_half = read_atran(path1)

    atran_40_second_half = read_atran(path2)

    atran_40 = pd.concat([atran_40_first_half, atran_40_second_half]).drop_duplicates(
        subset=["wav"]
    )
    return atran_40


def find_closest_atran_angle(zenith_angle):
    """
    Parses through the available ATRAN files and finds the one with the zenith angle closest to the desired.
    """
    files = glob("data/atran*obslat*")
    zenith_paths = np.array([eval(file.split("_")[1]) for file in files])
    closest_ind = np.argmin(abs(zenith_paths - zenith_angle))
    closest_zenith = zenith_paths[closest_ind]
    return closest_zenith


def add_tellurics_atran(
    wl_cube_model, flux_cube_model, n_order, n_exp, vary_airmass=False
):
    if vary_airmass:
        # assume now that I'm just using the same airmasses as before.
        zenith_angles = np.loadtxt("zenith_angles_w77ab.txt")

        # these are the beginning and ends of the two different ATRAN chunks.
        w_start_1 = 1.1
        w_end_1 = 2
        w_start_2 = 2
        w_end_2 = 3

        tell_splines = []  # will want one of these for each exposure.
        for zenith_angle in zenith_angles:
            zenith_path = find_closest_atran_angle(zenith_angle)
            path1 = f"data/atran_{zenith_path}_zenith_39_obslat_{w_start_1}_{w_end_1}_wave.dat"
            path2 = f"data/atran_{zenith_path}_zenith_39_obslat_{w_start_2}_{w_end_2}_wave.dat"
            atran_40 = read_atran_pair(path1, path2)
            tell_spline = interpolate.splrep(atran_40.wav, atran_40.depth, s=0.0)
            tell_splines += [tell_spline]

        for order in tqdm(range(n_order)):
            wl_grid = wl_cube_model[order]
            for exp in range(n_exp):
                tell_spline = tell_splines[exp]
                tell_flux = interpolate.splev(wl_grid, tell_spline, der=0)

                flux_cube_model[order, exp] *= tell_flux
    else:
        path1 = "atran_45k_1.3_2_microns_40_deg.txt"
        path2 = "atran_45k_2_2.7_microns_40_deg.txt"
        atran_40 = read_atran_pair(path1, path2)

        # todo: refactor. there are a lot of similarities!
        tell_spline = interpolate.splrep(atran_40.wav, atran_40.depth, s=0.0)
        for order in tqdm(range(n_order)):
            wl_grid = wl_cube_model[order]
            for exp in range(n_exp):
                tell_flux = interpolate.splev(wl_grid, tell_spline, der=0)
                flux_cube_model[order, exp] *= tell_flux

    return flux_cube_model


def calc_weighted_vector_insim(
    vals, eigenvector, relationships, provided_relationships=False
):
    """
    Calculates the weighted vector for a given set of eigenvalues and eigenvectors.

    Inputs
    ------
        :vals: (array) eigenvalues
        :eigenvector: (array) eigenvectors
        :relationships: (list of arrays) relationships between eigenvectors and the flux vector.
        :provided_relationships: (bool) if True, then relationships are provided. If False, then relationships are calculated.

    Outputs
    -------
        :fm: (array) weighted vector
        :relationships: (list of arrays) relationships between eigenvectors and the flux vector.
    """

    fm = np.zeros(1698)  # pad with 1s on all sides.
    if not provided_relationships:
        relationships = []
    for i, vector_ind in enumerate(range(4)):
        relationship = relationships[i]

        vector = eigenvector[:, vector_ind]

        eigenweight = np.dot(relationship, vals)

        weighted_vector = eigenweight * vector

        fm += weighted_vector
        if not provided_relationships:
            relationships += [relationship]
    fm[fm < 0.0] = 0.0
    return fm, relationships


def eigenweight_func(airmass, date):
    """
    Does the weighting. turns date and airmass into their correct array!

    Inputs
    ------
        :airmass: (float) airmass
        :date: (float) date

    Outputs
    -------
        :array: (array) array of values to be multiplied by the eigenvectors.
    """
    if airmass < 1.0 or airmass > 3.0:
        raise ValueError(
            f"Airmass must be valued between 1 amd 3. given airmass is {airmass}"
        )
    if date < -200 or date > 400:
        raise ValueError
    return np.array([1, airmass, date, date**2])


def add_tellurics(
    wl_cube_model,
    flux_cube_model,
    n_order,
    n_exp,
    vary_airmass=False,
    tell_type="ATRAN",
    time_dep_tell=False,
):
    """
    Includes tellurics in the model.
    todo: allow thew airmass variation to not be the case for the data-driven tellurics

    Inputs
    ------
        :wl_cube_model: (array) wavelength cube model
        :flux_cube_model: (array) flux cube model
        :n_order: (int) number of orders
        :n_exp: (int) number of exposures
        :vary_airmass: (bool) if True, then the airmass is varied. If False, then the airmass is fixed.
        :tell_type: (str) either 'ATRAN' or 'data-driven'
        :time_dep_tell: (bool) if True, then the tellurics are time dependent. If False, then the tellurics are not time dependent.

    Outputs
    -------
        :flux_cube_model: (array) flux cube model with tellurics included.

    """
    if tell_type == "data-driven":
        eigenvectors = np.load(abs_path + "/data/eigenvectors.npy")
        relationships_arr = np.load(abs_path + "/data/eigenweight_coeffs.npy")
        wave_for_eigenvectors = np.load(abs_path + "/data/wav_for_eigenvectors.npy")

        zenith_angles = np.loadtxt(abs_path + "/data/zenith_angles_w77ab.txt")

        airmasses = 1 / np.cos(np.radians(zenith_angles))  # todo: check units
        airmasses = airmasses[:n_exp]
        if time_dep_tell:
            # dates = np.loadtxt('dates_w77ab_scaled.txt')
            dates = np.linspace(0, 350, len(airmasses))
        else:
            dates = np.ones_like(airmasses)

        # iterate through the each exposure.
        for i, airmass in enumerate(airmasses):
            date = dates[i]
            for order in range(relationships_arr.shape[0]):
                vals = eigenweight_func(airmass, date)
                fm, _ = calc_weighted_vector_insim(
                    vals,
                    eigenvectors[order],
                    relationships_arr[order],
                    provided_relationships=True,
                )
                fm = np.concatenate([np.ones(150), fm])

                flux_cube_model[order][
                    i
                ] *= fm  # the orders don't quite line up. or maybe they do : )

    elif tell_type == "ATRAN":
        flux_cube_model = add_tellurics_atran(
            wl_cube_model, flux_cube_model, n_order, n_exp, vary_airmass=vary_airmass
        )
    return flux_cube_model
