import os
import tempfile

import numpy as np
import pytest

from scope.io import (  # Replace 'your_module' with the actual module name
    parse_input_file,
    write_input_file,
)


@pytest.fixture
def sample_files():
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as test_dir:
        # Create a sample input file
        input_file_content = """·········································
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
Author: Arjun Savel!
Planet: GJ 1214b

# Astrophysical Parameters
Rp                     1.5
Rp_solar               DATABASE
Rstar                  NULL
kp                     150.0
v_rot                  5.0
v_sys                  0.0

# Observation Parameters
observation            emission
phases                 0.0,0.25,0.5,0.75
blaze                  True
star                   False
"""
        input_file_path = os.path.join(test_dir, "test_input.txt")
        with open(input_file_path, "w") as f:
            f.write(input_file_content)

        # Create a sample database file
        db_content = """
planet_name,planet_radius_solar
TestPlanet,0.15
"""
        db_file_path = os.path.join(test_dir, "test_db.csv")
        with open(db_file_path, "w") as f:
            f.write(db_content)

        yield input_file_path, db_file_path


def test_parse_input_file(sample_files):
    input_file_path, db_file_path = sample_files
    data = parse_input_file(input_file_path, db_file_path)

    assert data["planet_name"] == "TestPlanet"
    assert data["Rp"] == 1.5
    assert data["Rp_solar"] == 0.15  # From database
    assert np.isnan(data["Rstar"])
    assert data["kp"] == 150.0
    assert data["v_rot"] == 5.0
    assert data["v_sys"] == 0.0
    assert data["observation"] == "emission"
    assert data["phases"] == [0.0, 0.25, 0.5, 0.75]
    assert data["blaze"] == True
    assert data["star"] == False


def test_write_input_file(sample_files, tmp_path):
    input_file_path, db_file_path = sample_files

    # First, parse the input file
    data = parse_input_file(input_file_path, db_file_path)

    # Write the data to a new file
    output_file_path = tmp_path / "output_input.txt"
    write_input_file(data, str(output_file_path))

    # Now parse the output file and compare with original data
    new_data = parse_input_file(str(output_file_path), db_file_path)

    # Compare key elements (you might want to expand this)
    assert new_data["planet_name"] == data["planet_name"]
    assert new_data["Rp"] == data["Rp"]
    assert new_data["Rp_solar"] == data["Rp_solar"]
    assert np.isnan(new_data["Rstar"])
    assert new_data["kp"] == data["kp"]
    assert new_data["observation"] == data["observation"]
    assert new_data["phases"] == data["phases"]
    assert new_data["blaze"] == data["blaze"]
    assert new_data["star"] == data["star"]


def test_parse_input_file_with_custom_params(sample_files):
    input_file_path, db_file_path = sample_files
    custom_param1 = 10
    custom_param2 = "value"

    data = parse_input_file(
        input_file_path,
        db_file_path,
        custom_param1=custom_param1,
        custom_param2=custom_param2,
    )

    assert data["custom_param1"] == custom_param1
    assert data["custom_param2"] == custom_param2
