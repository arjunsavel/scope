import numpy as np
import pandas as pd

# Mapping between input file parameters and database columns
parameter_mapping = {
    "Rp": "pl_radj",
    "Rstar": "st_rad",
    "v_sys": "system_velocity",
    "a": "pl_orbsmax",
    "P": "pl_orbtper",
    "v_sys": "st_radv",
}


def query_database(
    planet_name, parameter, database_path="data/default_params_exoplanet_archive.csv"
):
    try:
        df = pd.read_csv(database_path)
        # Use the mapped parameter name if it exists, otherwise use the original
        db_parameter = parameter_mapping.get(parameter, parameter)
        value = df.loc[df["planet_name"] == planet_name, db_parameter].values[0]
        return float(value)
    except Exception as e:
        print(f"Error querying database for {planet_name}, {parameter}: {e}")
        return np.nan


def parse_input_file(file_path, database_path="planet_database.csv"):
    # Read the file, skipping comment lines and empty lines
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        names=["parameter", "value"],
        comment="#",
        skip_blank_lines=True,
    )

    # Convert the dataframe to a dictionary
    data = dict(zip(df["parameter"], df["value"]))

    # Get the planet name (assuming it's in the input file)
    planet_name = data.get("planet_name", "")

    # List of astrophysical parameters
    astrophysical_params = ["Rp", "Rp_solar", "Rstar", "kp", "v_rot", "v_sys"]

    # Convert values to appropriate types
    for key, value in data.items():
        # Check for NULL
        if value == "NULL":
            data[key] = np.nan
        # Check for DATABASE in astrophysical parameters
        elif value == "DATABASE" and key in astrophysical_params:
            data[key] = query_database(planet_name, key, database_path)
        else:
            # Try to convert to float
            try:
                data[key] = float(value)
            except ValueError:
                # Check if it's a boolean
                if value.lower() == "true":
                    data[key] = True
                elif value.lower() == "false":
                    data[key] = False
                # Check if it's a list (for phases)
                elif "," in value:
                    data[key] = [float(v.strip()) for v in value.split(",")]
                # Otherwise, keep as string

    return data


def write_input_file(data, output_file_path="input.txt"):
    # Define the order and categories of parameters
    categories = {
        "Filepaths": ["planet_spectrum_path", "star_spectrum_path", "data_cube_path"],
        "Astrophysical Parameters": ["Rp", "Rp_solar", "Rstar", "kp", "v_rot", "v_sys"],
        "Instrument Parameters": ["SNR"],
        "Observation Parameters": [
            "observation",
            "phases",
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

    with open(output_file_path, "w") as f:
        # Write the header
        f.write(
            f"""·········································
:                                       :
:    ▄▄▄▄▄   ▄█▄    ████▄ █ ▄▄  ▄███▄   :
:   █     ▀▄ █▀ ▀▄  █   █ █   █ █▀   ▀  :
: ▄  ▀▀▀▀▄   █   ▀  █   █ █▀▀▀  ██▄▄    :
:  ▀▄▄▄▄▀    █▄  ▄▀ ▀████ █     █▄   ▄▀ :
:            ▀███▀         █    ▀███▀   :
:                           ▀           :
:                                       :
·········································
Created: 2024-08-15
Author: YourName
Planet: {data['planet_name']}

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

        # Write any remaining parameters that weren't in the predefined categories
        other_params = set(data.keys()) - set(sum(categories.values(), []))
        if other_params:
            f.write("# Other Parameters\n")
            for param in other_params:
                if param != "planet_name":  # We've already written this at the top
                    value = data[param]
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

    print(f"Input file written to {output_file_path}")


# Usage
write_input_file(input_data, "new_input.txt")


# Usage
input_data = parse_input_file("input.txt", "path/to/your/planet_database.csv")
print(input_data)
