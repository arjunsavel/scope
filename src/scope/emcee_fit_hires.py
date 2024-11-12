"""
Module to fit (simulated) HRCCS data with emcee, using MPI (multi-core).
The fit is with respect to the velocity parameters (Kp, Vsys) and the scale factor.
"""
import sys

import emcee
from schwimmbad import MPIPool  # todo: add these as dependencies...?

from scope.run_simulation import *

do_pca = True
np.random.seed(42)

# load the data
test_data_path = os.path.join(os.path.dirname(__file__), "../data")


def log_prob(
    x,
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
):
    """
    just add the log likelihood and the log prob.

    Inputs
    ------
        :x: (array) array of parameters

    Outputs
    -------
        :log_prob: (float) log probability.
    """
    rv_semiamp_orbit = 0.0
    Kp, Vsys, log_scale = x
    scale = np.power(10, log_scale)
    prior_val = prior(x, best_kp)

    if not np.isfinite(prior_val):
        return -np.inf
    ll = calc_log_likelihood(
        Vsys,
        Kp,
        scale,
        wl_cube_model,
        wl_model,
        Fp_conv,
        Fstar_conv,
        flux_cube,
        n_order,
        n_exposure,
        n_pixel,
        phases,
        Rp_solar,
        Rstar,
        rv_semiamp_orbit,
        A_noplanet,
        do_pca=do_pca,
        n_princ_comp=n_princ_comp,
        star=star,
        observation="transmission",
    )[0]
    return prior_val + ll


# @numba.njit
def prior(x, best_kp):
    """
    Prior on the parameters. Only uniform!

    Inputs
    ------
        :x: (array) array of parameters

    Outputs
    -------
        :prior_val: (float) log prior value.
    """
    Kp, Vsys, log_scale = x
    # do I sample in log_scale?
    if (
        best_kp - 50.0 < Kp < best_kp + 50.0
        and -50.0 < Vsys < 50.0
        and -1 < log_scale < 1
    ):
        return 0
    return -np.inf


def sample(
    nchains,
    nsample,
    A_noplanet,
    Fp_conv,
    wl_cube_model,
    n_order,
    n_exposure,
    n_pixel,
    star,
    n_princ_comp,
    flux_cube,
    wl_model,
    Fstar_conv,
    Rp_solar,
    Rstar,
    phases,
    do_pca=True,
    best_kp=192.06,
    best_vsys=0.0,
    best_log_scale=0.0,
):
    """
    Samples the likelihood. right now, it needs an instantiated best-fit value.

    Inputs
    ------
        :nchains: (int) number of chains
        :nsample: (int) number of samples
        :A_noplanet: (array) array of the no planet spectrum
        :fTemp: (array) array of the stellar spectrum
        :do_pca: (bool) whether to do PCA
        :best_kp: (float) best-fit planet velocity
        :best_vsys: (float) best-fit system velocity
        :best_log_scale: (float) best-fit log scale

    Outputs
    -------
        :sampler: (emcee.EnsembleSampler) the sampler object.
    """
    # todo: make the likelhiood function based on the sampling parameters.

    pos = np.array([best_kp, best_vsys, best_log_scale]) + 1e-2 * np.random.randn(
        nchains, 3
    )

    # Our 'pool' is just an object with a 'map' method which points to mpi_map
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        np.random.seed(42)

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob,
            args=(
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
            ),
            pool=pool,
        )
        sampler.run_mcmc(pos, nsample, progress=True)

    return sampler
