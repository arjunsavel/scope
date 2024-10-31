The input file
=======================

This document describes all available configuration parameters for the spectroscopic simulation pipeline.

Basic Information
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Created
     - Creation date of the configuration file. This can be ignored, really, because the this date will be overwritten by the current date when the simulation is run.
   * - Author
     - Author of the configuration file
   * - Planet name
     - Name of the target exoplanet. This field is used to resolve astrophysical parameters from the database if any of the parameters are set to DATABASE.

Simulation Setup
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - modelname
     - Name of the current model run. This sets the prefix and directory for the output files
   * - seed
     - Random number generator seed for noise generation. Set to None for a random seed. Different seeds will produce different photon noise realizations (under Poisson noise).

File Paths
---------
These parameters set the paths to the necessary files for the simulation.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - planet_spectrum_path
     - Path to the planetary spectrum file. This can be either a transmission or an emission spectrum; emission units are expected in W/m^2/micron.
   * - star_spectrum_path
     - Path to the stellar spectrum file. The example spectrum is a PHOENIX model, but any spectrum in the correct units can be used.
   * - data_cube_path
     - Path to the data cube file. This file sets the actual wavelength grid of the simulated data.

Astrophysical Parameters
----------------------
These parameters set the underlying simulated physical objects.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Rp
     - | Planetary radius in Jupiter radii
       | Can be set to DATABASE to pull from database
   * - Rstar
     - | Stellar radius in solar radii
       | Can be set to DATABASE to pull from database
   * - kp
     - | Expected planetary orbital velocity assuming circular orbit (km/s)
       | Set to NULL to calculate from orbital parameters
   * - v_sys
     - | Systemic velocity (km/s)
       | Can be set to DATABASE to pull from database
   * - v_rot
     - | Equatorial rotational velocity
       | Set to NULL to calculate from orbital parameters
   * - P_rot
     - | Orbital period of the planet (days). This can be used to calculated the equatorial rotation velocity if `v_rot` is set to `None`, assuming a spin-synchronized orbit.
       | Can be set to DATABASE to pull from database
   * - a
     - | Semi-major axis of the planet (AU). This can be used to calculated the planetary orbital velocity if `kp` is set to `NULL`.
       | Can be set to DATABASE to pull from database
   * - scale
     - Scaling factor for the model spectrum
   * - LD
     - | Enable/disable limb darkening
       | This parameter only affects transmission observations.
   * - u1
     - | First quadratic limb darkening coefficient
       | Not used if LD=False or in emission mode
   * - u2
     - | Second quadratic limb darkening coefficient
       | Not used if LD=False or in emission mode
   * - include_rm
     - | Enable/disable the Rossiter-McLaughlin effect
       | Only affects transmission observations
   * - v_rot_star
     - | Stellar equatorial rotational velocity (km/s)
       | Only used if include_rm=True in transmission mode
   * - lambda_misalign
     - | Misalignment angle between planet orbit and stellar rotation axis (degrees)
       | Only used if include_rm=True in transmission mode
   * - inc
     - | Orbital inclination relative to line of sight (degrees)
       | Only used if include_rm=True in transmission mode

Instrument Parameters
-------------------
These parameters set the instrumental setup and effects.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - blaze
     - Enable/disable the blaze function, which generally reduces throughput (and therefore increases noise) toward the edge of the orders.
   * - wav_error
     - Enable/disable wavelength solution errors. Included errors are fit from IGRINS WASP-77Ab data.
   * - order_dep_throughput
     - Enable/disable order-dependent throughput variations
   * - vary_throughput
     - | Enable/disable temporal throughput variations
       | IGRINS variations fit to WASP-77Ab emission data

Observation Parameters
--------------------
These parameters determine the type of simulated observations and the conditions under which they are taken.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - observation
     - | Observation type: "emission" or "transmission". This sets how the planetary and stellar signals are combined.
   * - phase_start
     - | Phase at observation start
       | 0 = transit center, 0.5 = secondary eclipse
       | Can be set to DATABASE to use transit duration
   * - phase_end
     - | Phase at observation end
       | 0 = transit center, 0.5 = secondary eclipse
       | Can be set to DATABASE to use transit duration
   * - n_exposures
     - Number of exposures to simulate
   * - star
     - Enable/disable stellar component
   * - telluric
     - Enable/disable telluric absorption.
   * - SNR
     - Signal-to-noise ratio per pixel
   * - tell_type
     - | Telluric simulation type
       | Options: "ATRAN" (radiative transfer, computed for IGRINS) or "data-driven" (fit to IGRINS standard star data)
   * - time_dep_tell
     - Enable/disable time-dependent tellurics. Only applicable when tell_type is set to "data-driven" and telluric is set to True.

Analysis Parameters
-----------------
These parameters set how the simulated data is analyzed to extract the planetary signal.
The only analysis currently implemented is Principal Components Analysis (PCA).

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - n_princ_comp
     - Number of principal components to remove before cross-correlation.
   * - divide_out_of_transit
     - | Enable/disable division by median out-of-transit data
       | Only used in transmission mode
   * - out_of_transit_dur
     - | Duration of out-of-transit data in units of transit duration
       | Only used if divide_out_of_transit=True in transmission mode
