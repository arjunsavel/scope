import pytest

from scope.emcee_fit_hires import *


@pytest.mark.parametrize(
    "values, output",
    [
        ([150, 0, 0], 0.0),
        ([2, 0, 0], -np.inf),
        ([150, -200, 0], -np.inf),
        ([150, 0, -100], -np.inf),
        ([150, -200, -100], -np.inf),
        ([2, 0, -100], -np.inf),
        ([150, -200, 0], -np.inf),
        ([2, -200, -100], -np.inf),
    ],
)
def test_prior(values, output):
    """
    Test the prior function.
    """
    Kp, Vsys, log_scale = values
    assert prior(values) == output
