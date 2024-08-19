import os
import tempfile

import numpy as np
import pytest

from scope.input_output import (  # Replace 'your_module' with the actual module name
    parse_input_file,
    write_input_file,
)


@pytest.fixture
def sample_files():
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as test_dir:
        # Create a sample input file
        input_file_content = """:········································
:                                       :
:    ▄▄▄▄▄   ▄█▄    ████▄ █ ▄▄  ▄███▄   :
:   █     ▀▄ █▀ ▀▄  █   █ █   █ █▀   ▀  :
: ▄  ▀▀▀▀▄   █   ▀  █   █ █▀▀▀  ██▄▄    :
:  ▀▄▄▄▄▀    █▄  ▄▀ ▀████ █     █▄   ▄▀ :
:            ▀███▀         █    ▀███▀   :
:                           ▀           :
:                                       :
:········································
Created:                2024-08-15
Author:                 Arjun Savel!
Planet name:            GJ 1214b

# Astrophysical Parameters
Rp                     1.5
Rp_solar               DATABASE
Rstar                  NULL
kp                     150.0
v_rot                  5.0
v_sys                  0.0

# Observation Parameters
observation            emission
phase_start                 0.3
phase_end                 0.5
blaze                  True
star                   False
"""
        input_file_path = os.path.join(test_dir, "test_input.txt")
        with open(input_file_path, "w") as f:
            f.write(input_file_content)

        # Create a sample database file
        db_content = """
pl_name,planet_radius_solar
GJ 1214b,0.15
"""
        db_file_path = os.path.join(test_dir, "test_db.csv")
        with open(db_file_path, "w") as f:
            f.write(db_content)

        yield input_file_path, db_file_path


def test_parse_input_file(sample_files):
    input_file_path, db_file_path = sample_files
    data = parse_input_file(input_file_path, db_file_path)

    assert data["planet_name"] == "GJ 1214b"
    assert data["Rp"] == 1.5
    assert np.isnan(data["Rstar"])
    assert data["kp"] == 150.0
    assert data["v_rot"] == 5.0
    assert data["v_sys"] == 0.0
    assert data["observation"] == "emission"
    assert data["phase_start"] == 0.3
    assert data["phase_end"] == 0.5
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
    assert np.isnan(new_data["Rstar"])
    assert new_data["kp"] == data["kp"]
    assert new_data["observation"] == data["observation"]
    assert new_data["phase_start"] == data["phase_start"]
    assert new_data["blaze"] == data["blaze"]
    assert new_data["star"] == data["star"]
