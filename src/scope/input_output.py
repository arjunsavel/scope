"""
Module to handle the input files.
"""

import argparse
import io
import os
import warnings
from datetime import datetime

import pandas as pd

from scope.calc_quantities import *
from scope.logger import *

logger = get_logger()


class ScopeConfigError(Exception):
    def __init__(self, message="scope input file error:"):
        self.message = message
        super().__init__(self.message)


# Mapping between input file parameters and database columns
parameter_mapping = {
    "Rp": "pl_radj",
    "Mp": "pl_bmassj",
    "Rstar": "st_rad",
    "Mstar": "st_mass",
    "v_sys": "system_velocity",
    "a": "pl_orbsmax",
    "P_rot": "pl_orbper",
    "v_sys": "st_radv",
    "planet_name": "pl_name",
    "lambda_misalign": "pl_projobliq",
    "e": "pl_orbeccen",
    "peri": "pl_orblper",
    "v_rot_star": "st_vsin",
    "b": "pl_imppar"
}


def query_database(
    planet_name, parameter, database_path="data/default_params_exoplanet_archive.csv"
):
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file {database_path} not found.")

    df = pd.read_csv(database_path)

    # check whether planet name is in database
    if planet_name not in df["pl_name"].values:
        logger.warning(f"Planet name {planet_name} not found in database.")
    try:
        # Use the mapped parameter name if it exists, otherwise use the original
        db_parameter = parameter_mapping.get(parameter, parameter)
        value = df.loc[df["pl_name"] == planet_name, db_parameter].values[0]
        return float(value)
    except Exception as e:
        logger.warning(f"Error querying database for {planet_name}, {parameter}: {e}")

        return np.nan


def unpack_lines(content):
    """
    Unpack lines from a file, removing comments and joining lines that are split.

    Parameters
    ----------
    content : list
        List of lines from the file.

    Returns
    -------
    data_lines : list
        List of lines with comments removed and split lines joined.
    """
    data_lines = []
    for line in content:
        line = line.strip()
        if line.startswith("Planet name:"):
            planet_name = line.split("Planet name:", 1)[1].strip()
        elif "Author:" in line:
            author = line.split("Author:", 1)[1].strip()
        elif (
            not line.startswith("#")
            and not line.startswith(":")
            and not line.startswith("Created")
            and line
        ):
            data_lines.append(line)
    return data_lines, planet_name, author


def coerce_nulls(data, key, value):
    if value == "NULL":
        data[key] = np.nan
        logger.warning(f" {key} is null on input")

    return data


def coerce_integers(data, key, value):
    integer_fields = ["n_exposures", "n_princ_comp", "seed"]
    if key in integer_fields:
        data[key] = int(value)
    else:
        try:
            data[key] = float(value)
        except:
            pass

    return data


def coerce_database(data, key, value, astrophysical_params, planet_name, database_path):
    
    if value == "DATABASE" and key in astrophysical_params:
        data[key] = query_database(planet_name, key, database_path)
        
    elif value == "DATABASE" and key in ["phase_start", "phase_end"]:
        tdur = query_database(planet_name, "pl_trandur", database_path)
        period = query_database(planet_name, "pl_orbper", database_path)

        # convert it to phase
        tdur_phase = convert_tdur_to_phase(tdur, period)

        if key == "phase_start":
            data[key] = -tdur_phase / 2
        else:
            data[key] = tdur_phase / 2

    return data


def coerce_splits(data, key, value):
    if isinstance(value, str) and "," in value:
        data[key] = [float(v.strip()) for v in value.split(",")]

    return data


def coerce_booleans(data, key, value):
    if value == "True":
        data[key] = True
    elif value == "False":
        data[key] = False

    return data


def parse_input_file(
    file_path, database_path="data/default_params_exoplanet_archive.csv", **kwargs
):
    """
    Parse an input file and return a dictionary of parameters.

    Parameters
    ----------
    file_path : str
        Path to the input file.
    database_path : str
        Path to the database file.
    **kwargs
        Additional keyword arguments to add to the data dictionary.

    Returns
    -------
    data : dict
        Dictionary of parameters.
    """
    # First, read the entire file content
    with open(file_path, "r") as file:
        content = file.readlines()

    data_lines, planet_name, author = unpack_lines(content)

    # Read the remaining lines with pandas
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=["parameter", "value"],
        comment="#",
    )

    # Convert the dataframe to a dictionary
    data = dict(zip(df["parameter"], df["value"]))

    # Add planet_name and author to the data dictionary
    data["planet_name"] = planet_name
    data["author"] = author

    # List of astrophysical parameters
    astrophysical_params = [
        "Rp",
        "Rp_solar",
        "Rstar",
        "kp",
        "v_rot",
        "v_sys",
        "P_rot",
        "a",
        "u1",
        "e",
        "b",
        "u2",
        "Mstar",
        "Mp",
        "peri",
    ]

    # Convert values to appropriate types
    for key, value in data.items():
        # check for values that are comma-delimited
        data = coerce_splits(data, key, value)

        # Check for NULL
        data = coerce_nulls(data, key, value)

        # make sure integers are cast as such
        data = coerce_integers(data, key, value)

        # make sure booleans are cast as such
        data = coerce_booleans(data, key, value)

        # Check for DATABASE in astrophysical parameters
        data = coerce_database(
            data, key, value, astrophysical_params, planet_name, database_path
        )

    # Add any additional kwargs to the data dictionary
    data.update(kwargs)

    data = calculate_derived_parameters(data)

    if data["tell_type"] == "data-driven" and data["blaze"] == False:
        raise ScopeConfigError("Data-driven tellurics requires blaze set to True.")

    return data


def write_input_file(data, output_file_path="input.txt"):
    # Define the order and categories of parameters

    categories = {
        "Filepaths": [
            "planet_spectrum_path",
            "star_spectrum_path",
            "data_cube_path",
            "snr_path",
        ],
        "Astrophysical Parameters": ["Rp", "Rp_solar", "Rstar", "kp", "v_rot", "v_sys"],
        "Instrument Parameters": ["SNR"],
        "Observation Parameters": [
            "observation",
            "phase_start",
            "phase_end",
            "n_exposures",
            "blaze",
            "star",
            "telluric",
            "tell_type",
            "time_dep_tell",
            "wav_error",
            "order_dep_throughput",
        ],
        "Analysis Parameters": ["n_princ_comp", "scale"],
    }
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file_path, "w") as f:
        # Write the header
        f.write(
            f""":·········································
:                                       :
:    ▄▄▄▄▄   ▄█▄    ████▄ █ ▄▄  ▄███▄   :
:   █     ▀▄ █▀ ▀▄  █   █ █   █ █▀   ▀  :
: ▄  ▀▀▀▀▄   █   ▀  █   █ █▀▀▀  ██▄▄    :
:  ▀▄▄▄▄▀    █▄  ▄▀ ▀████ █     █▄   ▄▀ :
:            ▀███▀         █    ▀███▀   :
:                           ▀           :
:                                       :
:········································
Created: {current_date}
Author: YourName
Planet name: {data['planet_name']}

""".format(
                date=pd.Timestamp.now().strftime("%Y-%m-%d")
            )
        )
        # Write parameters by category
        for category, params in categories.items():
            f.write(f"# {category}\n")
            for param in params:
                if param in data:
                    value = data[param]
                    # Handle different types of values
                    if isinstance(value, bool):
                        value = str(value)
                    elif isinstance(value, float):
                        if np.isnan(value):
                            value = "NULL"
                        else:
                            value = f"{value:.6f}".rstrip("0").rstrip(".")
                    elif isinstance(value, list):
                        value = ",".join(map(str, value))
                    f.write(f"{param:<23} {value}\n")
            f.write("\n")

    logger.info(f"Input file written to {output_file_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate observation")

    # Required parameters
    parser.add_argument(
        "--planet_spectrum_path", type=str, help="Path to planet spectrum"
    )
    parser.add_argument(
        "--star_spectrum_path", type=str, help="Path to star spectrum"
    )
    parser.add_argument(
        "--data_cube_path", type=str, help="Path to data cube"
    )

    # Optional parameters with their defaults matching your function
    parser.add_argument(
        "--phase_start",
        type=float,
        
        help="Start phase of the simulated observations",
    )
    parser.add_argument(
        "--phase_end",
        type=float,
        
        help="End phase of the simulated observations",
    )
    parser.add_argument(
        "--n_exposures", type=int, help="Number of exposures"
    )
    parser.add_argument(
        "--observation", type=str,  help="Observation type"
    )
    parser.add_argument("--blaze", type=bool, help="Blaze flag")
    parser.add_argument(
        "--n_princ_comp", type=int, help="Number of principal components"
    )
    parser.add_argument("--star", type=bool, help="Star flag")
    parser.add_argument("--SNR", type=float, help="Signal to noise ratio")
    parser.add_argument("--telluric", type=bool, help="Telluric flag")
    parser.add_argument(
        "--tell_type", type=str, help="Telluric type"
    )
    parser.add_argument(
        "--time_dep_tell", type=bool, help="Time dependent telluric"
    )
    parser.add_argument(
        "--wav_error", type=bool, help="Wavelength error flag"
    )
    parser.add_argument(
        "--rv_semiamp_orbit", type=float, help="RV semi-amplitude orbit"
    )
    parser.add_argument(
        "--order_dep_throughput",
        type=bool,
        
        help="Order dependent throughput",
    )
    parser.add_argument(
        "--Rp", type=float, help="Planet radius (Jupiter radii)"
    )
    parser.add_argument(
        "--Rstar", type=float, help="Star radius (solar radii)"
    )
    parser.add_argument(
        "--kp", type=float, help="Planetary orbital velocity (km/s)"
    )
    parser.add_argument("--v_rot", type=float, help="Rotation velocity")
    parser.add_argument("--scale", type=float, help="Scale factor")
    parser.add_argument("--v_sys", type=float, help="Systemic velocity")
    parser.add_argument(
        "--modelname", type=str, help="Model name"
    )
    parser.add_argument(
        "--divide_out_of_transit",
        type=bool,
        
        help="Divide out of transit",
    )
    parser.add_argument(
        "--out_of_transit_dur", type=float, help="Out of transit duration"
    )
    parser.add_argument(
        "--include_rm", type=bool, help="Include RM effect"
    )
    parser.add_argument(
        "--v_rot_star", type=float, help="Star rotation velocity"
    )
    parser.add_argument("--a", type=float, help="Semi-major axis")
    parser.add_argument(
        "--lambda_misalign", type=float, help="Misalignment angle"
    )
    parser.add_argument("--inc", type=float, help="Inclination")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--LD", type=bool, help="Limb darkening")
    parser.add_argument(
        "--vary_throughput", type=bool, help="Vary throughput"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    # For file input option
    parser.add_argument(
        "--input_file", type=str, default='input.txt',help="Input file with parameters"
    )

    return parser.parse_args()
