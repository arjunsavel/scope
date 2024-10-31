# scope
[![Documentation Status](https://readthedocs.org/projects/scope-astr/badge/?version=latest)](https://scope-astr.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CodeQL](https://github.com/arjunsavel/scope/actions/workflows/codeql.yml/badge.svg)](https://github.com/arjunsavel/scope/actions/workflows/codeql.yml)
[![Tests](https://github.com/arjunsavel/scope/actions/workflows/python-package.yml/badge.svg)](https://github.com/arjunsavel/scope/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/arjunsavel/scope/graph/badge.svg?token=2Q1NPQ4817)](https://codecov.io/gh/arjunsavel/scope)
[![Maintainability](https://api.codeclimate.com/v1/badges/d70a25a6766ee132bd94/maintainability)](https://codeclimate.com/github/arjunsavel/scope/maintainability)


Simulating cross-correlation of planetary emission.

# installation
You can install `scope` using pip:
```
pip install scope-astr
```


To install from source, run
```
python3 -m pip install -U pip
python3 -m pip install -U setuptools setuptools_scm pep517
git clone https://github.com/arjunsavel/scope
cd scope
python3 -m pip install -e .
```

You'll also need to download some data files. Currently, these data files are about 141 MB large. You can download them
(to the correct directory, even!) with the following:

```
cd src/scope
chmod +x download_data.bash
./download_data.bash
```

This will create a `data` directory and plop the relevant files into it. You're also welcome to run the tests to
make sure everything's been installed correctly:

```
pytest .
```

# workflow
For more details, see <a href="https://scope-astr.readthedocs.io/en/latest/">the documentation</a>.

Ideally, most user interaction with `scope` will simply occur through the input file (`scope/input.txt`).
Any data field in a row marked with the [DB] flag can be pulled from a local database by inputting `DATABASE`. In our case, a database simply refers
to a CSV containing contents from <a href="https://exoplanetarchive.ipac.caltech.edu/">the Exoplanet Archive</a>,
with planet parameters resolved with the planet name.

Simply edit the input file to the desired parameters, then run:

```
python run_simulation.py
```

Running the script requires exoplanet, stellar, and telluric spectra.
Default spectra and parameters currently correspond to the exoplanet WASP-77Ab.

Once completed, the code will create a directory for the data in `output/` (based on the input `modelname`)
with the following types of files:
- `simdata_`: the simulated flux cube with PCA performed. That is, the principal components with the largest variance have been removed.
- `nopca_simdata_`: the simulated flux cube, including all spectral components (exoplanet, star, blaze function, tellurics).
- `A_noplanet_`: the simulated flux cube with the *lowest-variance* principal component(s) removed.
- `lls_`: the log-likelihood surface for the simulated flux cube, as a Kp--Vsys map.
- `ccfs_`: the cross-correlation function for the simulated flux cube, as a Kp--Vsys map.
