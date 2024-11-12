"""
An example parameter grid.
"""

import numpy as np
from sklearn.model_selection import ParameterGrid

blazes = [False, True]
order_dep_throughputs = [False]
tellurics = [False, True]
telluric_types = ["ATRAN"]
time_dep_tellurics = [False]
wav_errors = [False]
stars = [False, True]

NPCs = np.arange(5)

SNRs = [0, 60, 250]

parameter_list1 = list(
    ParameterGrid(
        {
            "blaze": blazes,
            "n_princ_comp": NPCs,
            "SNR": SNRs,
            "star": stars,
            "telluric": tellurics,
            "wav_error": wav_errors,
            "time_dep_telluric": time_dep_tellurics,
            "telluric_type": telluric_types,
            "order_dep_throughput": order_dep_throughputs,
        }
    )
)

"""
second grid — needs to include the new parameters.
    - new tellurics
    - new tellurics with time
    - wavelength-dependent throughput
    - Doppler shift jitter.

just make sure that I start indexing at the new one :)
"""


blazes = [True]

order_dep_throughputs = [True, False]

tellurics = [False, True]
telluric_types = ["data-driven"]
time_dep_tellurics = [False, True]
stars = [False, True]
wav_errors = [False, True]

NPCs = np.arange(5)

SNRs = [0, 60, 250]

parameter_list2 = list(
    ParameterGrid(
        {
            "blaze": blazes,
            "n_princ_comp": NPCs,
            "SNR": SNRs,
            "star": stars,
            "telluric": tellurics,
            "wav_error": wav_errors,
            "time_dep_telluric": time_dep_tellurics,
            "telluric_type": telluric_types,
            "order_dep_throughput": order_dep_throughputs,
        }
    )
)

parameter_list = parameter_list1 + parameter_list2
