"""
Stores constants used in the scope package. We store the values themselves so that jax and numba need not have access
to astropy or scipy.
"""
import astropy.constants as const

const_c = const.c.si.value
rjup_rsun = (const.R_jup / const.R_sun).si.value
rsun = const.R_sun.si.value
