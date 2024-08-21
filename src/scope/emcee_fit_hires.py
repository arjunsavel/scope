import pickle
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
    Kp, Vsys, log_scale = x
    scale = np.power(10, log_scale)
    prior_val = prior(x, best_kp)
    if not np.isfinite(prior_val):
        return -np.inf
    return (
        prior_val
        + calc_log_likelihood(
            Vsys,
            Kp,
            scale,
            wl_cube_model,
            Fp_conv,
            n_order,
            n_exposure,
            n_pixel,
            A_noplanet=A_noplanet,
            do_pca=do_pca,
            n_princ_comp=n_princ_comp,
            star=star,
        )[0]
    )


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
                do_pca,
            ),
            pool=pool,
        )
        sampler.run_mcmc(pos, nsample, progress=True)

    return sampler


if __name__ == "__main__":
    # example below!

    # ind = eval(sys.argv[1])
    ind = 111  # SNR = 60, blaze, tell, star, 4 PCA components
    param_dict = parameter_list[ind]

    blaze, NPC, star, SNR, telluric = (
        param_dict["blaze"],
        param_dict["n_princ_comp"],
        param_dict["star"],
        param_dict["SNR"],
        param_dict["telluric"],
    )

    lls, ccfs = np.zeros((50, 50)), np.zeros((50, 50))

    A_noplanet, fTemp, fTemp_nopca, just_tellurics = make_data(
        1.0,
        wl_cube_model,
        do_pca=True,
        blaze=blaze,
        n_princ_comp=NPC,
        tellurics=telluric,
        Vsys=0,
        star=star,
        Kp=192.06,
        SNR=SNR,
    )

    with open("mcmc_pre_pca.pkl", "wb") as f:
        pickle.dump(fTemp_nopca, f)
    nchains = 26
    nsample = 5000
    sampler = sample(nchains, nsample, A_noplanet, fTemp, do_pca=True)
    with open("mcmc_samples.pkl", "wb") as f:
        pickle.dump(sampler, f)
    # np.savetxt('output/mcmc_samples.txt', samples)
