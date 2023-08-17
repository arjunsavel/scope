import pickle
import sys

import emcee
from run_simulation import *
from schwimmbad import MPIPool  # todo: add these as dependencies...?
from utils import *

do_pca = True
np.random.seed(42)


def log_prob(x):
    """
    just add the log likelihood and the log prob.
    """
    Kp, Vsys, log_scale = x
    scale = np.power(10, log_scale)
    prior_val = prior(x)
    if not np.isfinite(prior_val):
        return -np.inf
    return (
        prior_val
        + calc_log_likelihood(
            Vsys,
            Kp,
            scale,
            wl_cube_model,
            fTemp,
            Ndet,
            n_exposure,
            n_pixel,
            A_noplanet=A_noplanet,
            do_pca=do_pca,
            NPC=n_princ_comp,
            star=star,
        )[0]
    )


# van Sluijs+22 fit for log10 a. so do Brogi et al.
# @numba.njit
def prior(x):
    Kp, Vsys, log_scale = x
    # do I sample in log_scale?
    if 146.0 < Kp < 246.0 and -50 < Vsys < 50 and -1 < log_scale < 1:
        return 0
    return -np.inf


def sample(
    nchains,
    nsample,
    A_noplanet,
    fTemp,
    do_pca=True,
    best_kp=192.06,
    best_vsys=0.0,
    best_log_scale=0.0,
):
    """
    Samples the likelihood. right now, it needs an instantiated best-fit value.
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

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        sampler.run_mcmc(pos, nsample, progress=True)

    return sampler


if __name__ == "__main__":
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

    # redoing the grid. how close does PCA get to a tellurics-free signal detection?
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
