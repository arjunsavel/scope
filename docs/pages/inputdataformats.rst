Input Data Formats
====================

The simulation pipeline requires three input files: a planet spectrum, a stellar spectrum, and a data cube. This document describes the format and requirements for each file.

Planet Spectrum
-------------

**File Format**: Pickle (.pic)

**Array Shape**: (2 x n_wavelength)

The planet spectrum file contains two sub-arrays:

1. Wavelength grid (in microns)
2. Planet spectrum, with units depending on observation type:

   * **Transmission observations**: Transit depth, defined as (Rp / Rstar)²
   * **Emission observations**: Spectral flux density in W/m²/m

Stellar Spectrum
--------------

**File Format**: Text (.txt)

**Structure**: Two-column format with no headers:

1. Column 1: Wavelength (microns)
2. Column 2: Spectral flux density (W/m²/m)

Data Cube
--------

**File Format**: Pickle (.pic)

**Contents**: Contains two primary arrays:

1. Wavelength grid:

   * Shape: (n_orders x n_wavelength)
   * Units: microns
   * Purpose: Sets the wavelength grid for the simulation

2. Flux data:

   * Shape: (n_orders x n_exposures x n_wavelengths)
   * Note: This data is not used in the simulation but is included in the file structure

.. note::
   The data cube is based on real observational data and defines the wavelength grid used throughout the simulation pipeline.
