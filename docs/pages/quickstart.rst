Quickstart
----------
The aim of `scope` is to simulate observations of exoplanet atmospheres with ground-based, high-resolution spectroscopy.
Simulating the atmosphere *itself* — performing the radiative transfer calculations to generate a spectrum
that depends on atmospheric properties — is out of scope for `scope`.

Once the package is installed and data are downloaded, the simulation is as simple as running, from the command line:

.. code-block:: python

    python run_simulation.py



`scope` can also be run within a Python interpreter. The following code snippet demonstrates how to run a simulation
(note that these are not real filepaths!):

.. code-block:: python

    from scope import run_simulation

    simulate_observation(
    planet_spectrum_path="./planet_spectrum.pkl",
    star_spectrum_path="./stellar_spectrum.pkl",
    data_cube_path="./data_cube.pkl",
