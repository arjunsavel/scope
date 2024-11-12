"""
Module to handle the input files.
"""

import io
import os
from datetime import datetime

import pandas as pd

from scope.calc_quantities import *


class ScopeConfigError(Exception):
    def __init__(self, message="scope input file error:"):
        self.message = message
        super().__init__(self.message)


# Mapping between input file parameters and database columns
parameter_mapping = {
    "Rp": "pl_radj",
    "Rstar": "st_rad",
    "v_sys": "system_velocity",
    "a": "pl_orbsmax",
    "P_rot": "pl_orbper",
    "v_sys": "st_radv",
    "planet_name": "pl_name",
    "Rp_solar": "planet_radius_solar",
    "lambda_misalign": "pl_projobliq",
}


def query_database(
    planet_name, parameter, database_path="data/default_params_exoplanet_archive.csv"
):
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file {database_path} not found.")

    df = pd.read_csv(database_path)

    # check whether planet name is in database
    if planet_name not in df["pl_name"].values:
        print(f"Planet name {planet_name} not found in database.")
    try:
        # Use the mapped parameter name if it exists, otherwise use the original
        db_parameter = parameter_mapping.get(parameter, parameter)
        value = df.loc[df["pl_name"] == planet_name, db_parameter].values[0]
        return float(value)
    except Exception as e:
        print(f"Error querying database for {planet_name}, {parameter}: {e}")

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

    return data


def coerce_integers(data, key, value):
    integer_fields = ["n_exposures", "n_princ_comp"]
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
        "u2",
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
        "Filepaths": ["planet_spectrum_path", "star_spectrum_path", "data_cube_path"],
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

    print(f"Input file written to {output_file_path}")
