import pytest

from scope.emcee_fit_hires import *
from scope.tests.conftest import test_baseline_outouts, test_inputs

test_data_path = os.path.join(os.path.dirname(__file__), "../data")
do_pca=True

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
    assert prior(values, best_kp=160.0) == output


def test_likelihood(test_baseline_outouts, test_inputs):
    (
        A_noplanet,
        flux_cube,
        flux_cube_nopca,
        just_tellurics,
        n_exposure,
        n_order,
        n_pixel,
    ) = test_baseline_outouts
    star = True
    Rp_solar = 0.1
    Rstar = 1.0
    phases = np.linspace(-0.01, 0.01, n_exposure)
    Fp_conv, Fstar_conv, wl_cube_model, wl_model = test_inputs
    best_kp = 192.06

    x_good = [best_kp, 0, 0]
    x_bad = [144, -50, -0.9]
    n_princ_comp = 4
    log_prob_good = log_prob(
        x_good,
        best_kp,
        wl_cube_model,
        Fp_conv,
        n_order,
        n_exposure,
        n_pixel,
        A_noplanet,
        star,
        n_princ_comp,
        flux_cube,
        wl_model,
        Fstar_conv,
        Rp_solar,
        Rstar,
        phases,
        do_pca,
    )

    log_prob_bad = log_prob(
        x_bad,
        best_kp,
        wl_cube_model,
        Fp_conv,
        n_order,
        n_exposure,
        n_pixel,
        A_noplanet,
        star,
        n_princ_comp,
        flux_cube,
        wl_model,
        Fstar_conv,
        Rp_solar,
        Rstar,
        phases,
        do_pca,
    )

    assert log_prob_good > log_prob_bad
