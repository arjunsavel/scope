# scope
[![Documentation Status](https://readthedocs.org/projects/scope-astr/badge/?version=latest)](https://scope-astr.readthedocs.io/en/latest/?badge=latest) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CodeQL](https://github.com/arjunsavel/scope/actions/workflows/codeql.yml/badge.svg)](https://github.com/arjunsavel/scope/actions/workflows/codeql.yml)
[![Tests](https://github.com/arjunsavel/scope/actions/workflows/python-package.yml/badge.svg)](https://github.com/arjunsavel/scope/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/arjunsavel/scope/graph/badge.svg?token=2Q1NPQ4817)](https://codecov.io/gh/arjunsavel/scope)


Simulating cross-correlation of planetary emission.

# installation
To install from source, run
```
python3 -m pip install -U pip
python3 -m pip install -U setuptools setuptools_scm pep517
git clone https://github.com/arjunsavel/scope
cd scope
python3 -m pip install -e .
```

# workflow
The bulk of `scope`'s high-level functionality is contained in `scope/run_simulation.py`.
Running the file with `python run_simulation.py n` will run a simulation in the defined grid at index `n`. 
The grid itself is laid out in `scope/run_simulation.py`.

Running the script requires an exoplanet spectrum, stellar spectrum, and telluric spectrum.
Default parameters are currently correspond to the exoplanet WASP-77Ab.

Once completed, the code will output:
- `simdata` file: the simulated flux cube with PCA performed. That is, the principle components with the largest variance have been removed.
- `nopca_simdata` file: the simulated flux cube, including all spectral components (exoplanet, star, blaze function, tellurics).
- `A_noplanet` file: the simulated flux cube with the *lowest-variance* principle component removed.
- `lls_` file: the log-likelihood surface for the simulated flux cube, as a Kp--Vsys map.
- `ccfs_` file: the cross-correlation function for the simulated flux cube, as a Kp--Vsys map.
