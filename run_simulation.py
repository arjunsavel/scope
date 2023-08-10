import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import splev, splrep
import astropy.constants as const
import astropy.io.fits as fits
import pdb
from numba import njit
from hires_obs_simulator import *
from sklearn.model_selection import ParameterGrid
import sys
np.random.seed(42)
const_c = constants.c
Rp = 1.21  # in Rjup
Rstar = 0.955  # in Rsun
Kp = 192.02


def make_data(
    scale,
    wlgrid,
    do_pca=True,
    blaze=False,
    do_airmass_detrending=False,
    tellurics=False,
    NPC=4,
    Vsys=0,
    Kp=192.06,
    star=False,
    SNR=0,
    observation='emission',
    tell_type='ATRAN',
    time_dep_tell=False,
    wav_error=False,
    order_dep_throughput=False
):
    """
    Creates a simulated HRCCS dataset. Main function.

    Inputs
    ------
        :scale: (float) scaling factor for the data.
        :wlgrid: (array) wavelength grid for the data.
        :do_pca: (bool) if True, do PCA on the data.
        :blaze: (bool) if True, include the blaze function in the data.
        :do_airmass_detrending: (bool) if True, do airmass detrending on the data.
        :tellurics: (bool) if True, include tellurics in the data.
        :NPC: (int) number of principal components to use.
        :Vsys: (float) systemic velocity of the planet.
        :Kp: (float) semi-amplitude of the planet.
        :star: (bool) if True, include the stellar spectrum in the data.
        :SNR: (float) photon signal-to-noise ratio (not of the planetary spectrum itself!)
        :observation: (str) 'emission' or 'transmission'.
        :tell_type: (str) 'ATRAN' or 'data-driven'.
        :time_dep_tell: (bool) if True, include time-dependent tellurics. not applicable for ATRAN.
        :wav_error: (bool) if True, include wavelength-dependent error.
        :order_dep_throughput: (bool) if True, include order-dependent throughput.

    Outputs
    -------
        :A_noplanet: (array) the data cube with only the larger-scale trends.
        :fTemp: (array) the data cube with the larger-scale trends removed.
        :fTemp_nopca: (array) the data cube with all components.
        :just_tellurics: (array) the telluric model that's multiplied to the dataset.

    """
    Kstar = 0.3229 * 1.0

    Rvel = 1.6845

    RV = (
        Vsys + Rvel + Kp * np.sin(2.0 * np.pi * ph)
    )  # Vsys is an additive term around zero
    dl_l2 = RV * 1e3 / const_c
    RVs = (Vsys + Rvel - Kstar * np.sin(2.0 * np.pi * ph))

    # Looping through each phase and computing total log-L by summing logLs for each obvservation/phase
    fTemp = np.zeros((Ndet, Nphi, Npix))
    A_noplanet = np.zeros((Ndet, Nphi, Npix))
    for j in range(Ndet):
        wCut = np.copy(
            wlgrid[
                j,
            ]
        )  # Cropped wavelengths
        for i in range(Nphi):
            wShift = wCut * (1.0 - dl_l2[i])
            wls = wCut * (1.0 - RVs[i] * 1e3 / constants.c)

            # do keep in mind that Fp is actually *flux* in emission, but in transmission, it's the absorption in fractional amounts.
            Fp = np.interp(wShift, wl_model, Fp_conv * (Rp * 0.1) ** 2) * scale
            Fs = np.interp(wls, wl_model, Fstar_conv * (Rstar) ** 2)
            if star:
                if observation == 'emission':
                    fTemp[j, i,] = (
                        #Fp / Fs + 1.0
                        Fp + Fs # now want to get the star in there
                    )
                elif observation == 'transmission':
                    fTemp[j, i,] = (
                        # Fp / Fs + 1.0
                            Fs * (1 - Fp)  # now want to get the star in there
                    )
            else:
                fTemp[
                    j,
                    i,
                ] = Fp

    # ok now do the PCA. where does it fall apart?
    # todo: think about if and where I add noise.
    if tellurics:
        fTemp = add_tellurics(wl_cube_model, fTemp, Ndet, Nphi, vary_airmass=True, tell_type=tell_type,
                              time_dep_tell=time_dep_tell)
        # these spline fits aren't perfect. we mask negative values to 0.
        just_tellurics = np.ones_like(fTemp)
        just_tellurics = add_tellurics(wl_cube_model, just_tellurics, Ndet, Nphi, vary_airmass=True, tell_type=tell_type,
                                       time_dep_tell=time_dep_tell)
        fTemp[fTemp < 0.] = 0.0
        just_tellurics[just_tellurics < 0.0] = 0.0
    if blaze:
        fTemp = add_blaze_function(wl_cube_model, fTemp, Ndet, Nphi)
        fTemp[fTemp < 0.] = 0.0
    fTemp = detrend_cube(
        fTemp, Ndet, Nphi
    )  # TODO: think about mean subtraction vs. division here? also, this is basically the wavelength-calibration step!
    if SNR > 0:  # 0 means don't add noise!
        if order_dep_throughput:
            noise_model = 'IGRINS'
        else:
            noise_model = 'constant'
        fTemp = add_noise_cube(fTemp, wl_cube_model, SNR, noise_model=noise_model)

    if wav_error:
        doppler_shifts = np.loadtxt('data/doppler_shifts_w77ab.txt') # todo: create this!
        fTemp = change_wavelength_solution(wl_cube_model, fTemp, doppler_shifts)
    fTemp_nopca = fTemp.copy()
    if do_pca:
        for j in range(Ndet):
            # todo: redo all analysis centering on 0
            u, s, vh = np.linalg.svd(fTemp[j], full_matrices=False)  # decompose
            s_noplanet = s.copy()
            s_noplanet[NPC:] = 0.0
            W = np.diag(s_noplanet)
            A_noplanet[j] = np.dot(u, np.dot(W, vh))

            s[:NPC] = 0.0
            W = np.diag(s)
            fTemp[j] = np.dot(u, np.dot(W, vh))
    elif do_airmass_detrending:
        zenith_angles = np.loadtxt('data/zenith_angles_w77ab.txt')
        airm = 1/np.cos(np.radians(zenith_angles)) # todo: check units 
        polyDeg = 2.  # Setting the degree of the polynomial fit
        xvec = airm
        fAirDetr = np.zeros((Ndet, Nphi, Npix))
        for io in range(Ndet):
            # Now looping over columns
            for i in range(Npix):
                yvec = fTemp[io,:,i].copy()
                fit = np.poly1d(np.polyfit(xvec,yvec,polyDeg))(xvec)
                fAirDetr[io,:,i] = fTemp[io,:,i] / fit
                A_noplanet[io,:,i] = fit
        fTemp = fAirDetr
        fTemp[~np.isfinite(fTemp)] = 0.
    else:
        for j in range(Ndet):
            for i in range(Nphi):
                fTemp[j][i] -= np.mean(fTemp[j][i])
            
            
    if np.all(A_noplanet == 0):
        print("was all zero")
        A_noplanet = np.ones_like(A_noplanet)
    if tellurics:
        return A_noplanet, fTemp, fTemp_nopca, just_tellurics
    return A_noplanet, fTemp, fTemp_nopca, np.ones_like(fTemp)  # returning CCF and logL values
# every fTemp that's returned is going to be centered on 0. so how does that telluric removal work?

#@njit
def log_likelihood_PCA(
    Vsys,
    Kp,
    scale,
    wlgrid,
    fTemp,
    Ndet,
    Nphi,
    Npix,
    A_noplanet=1,
    do_pca=False,
    NPC=4,
    star=False,
    observation='emission',
        #cheat_tellurics=False,
        #input_tellurics=[1] # this is the same as the previous parameter. how to set this?
):
    Kstar = 0.3229 * 1.0

    # Kstar=(Mp/Mstar*9.55E-4)*Kp  #this is mass planet/mass star

    I = np.ones(Npix)
    N = Npix  # np.array([Npix])
    Rvel = 1.6845

    # Time-resolved total radial velocity
    RV = (
        Vsys + Rvel + Kp * np.sin(2.0 * np.pi * ph)
    )  # Vsys is an additive term around zero
    dl_l = RV * 1e3 / const_c

    RVs = (Vsys + Rvel - Kstar * np.sin(2.0 * np.pi * ph))

    # Initializing log-likelihoods and CCFs
    logL_Matteo = 0.0
    CCF = 0.0
    # Looping through each phase and computing total log-L by summing logLs for each obvservation/phase

    # ok...if I didn't remove any PCA components...then I can just keep it normal? just multiply it by ones!!!!!!!!!
    for j in range(Ndet):
        wCut = np.copy(
            wlgrid[
                j,
            ]
        )  # Cropped wavelengths
        gTemp = np.zeros((Nphi, Npix))  # "shifted" model spectra array at each phase
        for i in range(Nphi):
            wShift = wCut * (1.0 - dl_l[i])
            Fp = np.interp(wShift, wl_model, Fp_conv * (Rp * 0.1) ** 2) * scale
            wls = wCut * (1.0 - RVs[i] * 1e3 / constants.c)
            Fs = np.interp(wls, wl_model, Fstar_conv * (Rstar) ** 2)

            if star and  observation == 'emission':
                gTemp[i,] = (
                    Fp / Fs + 1.0
                )
            else: # in transmission, after we "divide out" (with PCA) the star and tellurics, we're left with Fp...kinda
                gTemp[
                    i,
                ] = (1 - Fp)

        # ok now do the PCA. where does it fall apart?
        if do_pca:
            # process the model same as the "data"!
            gTemp *= A_noplanet[j]
            u, ss, vh = np.linalg.svd(gTemp, full_matrices=False)  # decompose
            ss[:NPC] = 0.0
            W = np.diag(ss)
            gTemp = np.dot(u, np.dot(W, vh))
        elif do_airmass_detrending:
            # re-processing as if it were PCA. not really any big difference here, likely, but just in case!
            gTemp *= A_noplanet[j]
            zenith_angles = np.loadtxt('data/zenith_angles_w77ab.txt')
            airm = 1/np.cos(np.radians(zenith_angles)) # todo: check units                                                                                                                                             
            polyDeg = 2.  # Setting the degree of the polynomial fit                                                                                                                                                   
            xvec = airm
            fAirDetr = np.zeros((Ndet, Nphi, Npix))
            for io in range(Ndet):
                 # Now looping over columns                                                                                                                                                                             
                for i in range(Npix):
                    yvec = fContNorm[io,:,i].copy()
                    fit = np.poly1d(np.polyfit(xvec,yvec,polyDeg))(xvec)
                    fAirDetr[io,:,i] = gTemp[io,:,i] / fit
            gTemp = fAirDetr
            
            gTemp[~np.isfinite(gTemp)] = 0.
            
        for i in range(Nphi):
            gVec = np.copy(
                gTemp[
                    i,
                ]
            )
            gVec -= (gVec.dot(I)) / float(Npix)  # mean subtracting here...
            sg2 = (gVec.dot(gVec)) / float(Npix)

            fVec = np.copy(
                fTemp[j][
                    i,
                ]
            )  # already mean-subtracted! needed for the previous PCA! or...can I do the PCA without normalizing first? TODO
            # fVec-=(fVec.dot(I))/float(Npix)
            sf2 = (fVec.dot(fVec)) / float(Npix)

            R = (fVec.dot(gVec)) / float(Npix)  # cross-covariance
            CC = R / np.sqrt(sf2 * sg2)  # cross-correlation
            CCF += CC
            logL_Matteo += -0.5 * N * np.log(sf2 + sg2 - 2.0 * R)

    return logL_Matteo, CCF  # returning CCF and logL values


Kparr = np.linspace(93.06, 292.06, 200)
Vsys_all = np.arange(-100, 100)
Ndet, Nphi, Npix = (44, 79, 1848)
scale = 1.0
with open("data_RAW_20201214_wl_algn_03.pic", "rb") as f:
    mike_wave, mike_cube = pickle.load(f, encoding="latin1")

wl_cube_model = mike_wave.copy().astype(np.float64)

blazes = [False, True]
order_dep_throughputs = [False]
tellurics = [False, True]
telluric_types = ['ATRAN']
time_dep_tellurics = [False]
wav_errors = [False]
stars = [False, True]

NPCs = np.arange(5)

SNRs = [0, 60, 250]

parameter_list1 = list(ParameterGrid({'blaze': blazes,
    'NPC':NPCs,
    'SNR':SNRs,
                                     'star': stars,
                                      'telluric':tellurics,
                                      'wav_error': wav_errors,
                                      'time_dep_telluric': time_dep_tellurics,
                                      'telluric_type': telluric_types,
                                      'order_dep_throughput': order_dep_throughputs}))


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
telluric_types = ['data-driven']
time_dep_tellurics = [False, True]
stars = [False, True]
wav_errors = [False, True]

NPCs = np.arange(5)

SNRs = [0, 60, 250]

parameter_list2 = list(ParameterGrid({'blaze': blazes,
    'NPC':NPCs,
    'SNR':SNRs,
                                     'star': stars,
                                    'telluric':tellurics,
                                      'wav_error': wav_errors,
                                      'time_dep_telluric': time_dep_tellurics,
                                      'telluric_type': telluric_types,
                                      'order_dep_throughput': order_dep_throughputs}))

parameter_list = parameter_list1 + parameter_list2


ph = pickle.load(open("ph.pic", "rb"), encoding="latin1")  # Time-resolved phases
Rvel = pickle.load(
    open("rvel.pic", "rb"), encoding="latin1"
)  # +30.  # Time-resolved Earth-star velocity

# so if I want to change my model, I just alter this!
wl_model, Fp, Fstar = pickle.load(
    open("data/best_fit_spectrum.pic", "rb"), encoding="latin1"
)  # the model you generated in call_pymultinest.py
wl_model, Fp_model, Fstar = pickle.load(
    open("data/best_fit_spectrum.pic", "rb"), encoding="latin1"
)  # the model you generated in call_pymultinest.py   
wl_model = wl_model.astype(np.float64)
#"/mnt/home/asavel/CODE_FOR_PAPER/RETRIEVAL/FIDUCIAL_RUN/best_fit_spectrum_just_hcn_-1_dex.pic"

##rotational coonvolutiono
vsini = 4.5
ker_rot = get_rot_ker(vsini, wl_model)
Fp_conv_rot = np.convolve(Fp, ker_rot, mode="same")
Fp_conv_rot_model = np.convolve(Fp_model, ker_rot, mode="same")

# instrument profile convolustion
xker = np.arange(41) - 20
sigma = 5.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # nominal
yker = np.exp(-0.5 * (xker / sigma) ** 2.0)
yker /= yker.sum()
Fp_conv = np.convolve(Fp_conv_rot, yker, mode="same")
Fp_conv_model = np.convolve(Fp_conv_rot_model, yker, mode="same")
# Fstar_conv=np.convolve(Fstar,yker,mode='same')


star_wave, star_flux = np.loadtxt("data/PHOENIX_5605_4.33.txt").T
Fstar_conv = get_star_spline(star_wave, star_flux, wl_model, yker, smooth=False)


if __name__ == "__main__":

    ind = eval(sys.argv[1])
    param_dict = parameter_list[ind]
    
    blaze, NPC, star, SNR, telluric, tell_type, time_dep_tell, wav_error, order_dep_throughput = param_dict['blaze'],\
                                                                                                param_dict['NPC'], \
                                                                                                param_dict['star'], \
                                                                                                param_dict['SNR'], \
                                                                                                param_dict['telluric'], \
                                                                                                param_dict['telluric_type'], \
                                                                                                 param_dict['time_dep_telluric'], \
                                                                                                 param_dict['wav_error'], \
                                                                                                param_dict['order_dep_throughput']
    
    lls, ccfs = np.zeros((50,50)), np.zeros((50,50))


    # redoing the grid. how close does PCA get to a tellurics-free signal detection?
    A_noplanet, fTemp, fTemp_nopca, just_tellurics = make_data(scale,
                      wl_cube_model, 
                      do_pca=False,
                      do_airmass_detrending=True,
                      blaze=blaze, 
                      NPC=NPC,
                      tellurics=telluric, 
                      Vsys=0, 
                      star=star,
                      Kp=192.06,
                      SNR=SNR,
                      tell_type=tell_type,
                      time_dep_tell=time_dep_tell,
                      wav_error=wav_error,
                      order_dep_throughput=order_dep_throughput)
    with open(f'output/simdata_{NPC}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt', 'wb') as f:
        pickle.dump(fTemp, f)
    with open(f'output/nopca_simdata_{NPC}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt', 'wb') as f:
        pickle.dump(fTemp_nopca, f)
    with open(f'output/A_noplanet_{NPC}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt', 'wb') as f:
        pickle.dump(A_noplanet, f)

    # only save if tellurics are True. Otherwise, this will be cast to an array of ones.
    if tellurics:
        with open(f'output/just_tellurics_vary_airmass.txt', 'wb') as f:
            pickle.dump(just_tellurics, f)
    for i, KP in tqdm(enumerate(Kparr[50:150][::2]), total=50, desc='looping PCA over Kp'):
        for j, vsys in enumerate(Vsys_all[50:150][::2]):
            res = log_likelihood_PCA(vsys, KP, scale, wl_cube_model, fTemp, Ndet, Nphi, Npix,
                                    do_pca=True, NPC=NPC,A_noplanet=A_noplanet ,star=star)
            lls[i,j] = res[0] 
            ccfs[i,j] = res[1]
    print(f'output/lls_{NPC}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt')
    np.savetxt(f'output/lls_{NPC}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt', lls)
    np.savetxt(f'output/ccfs_{NPC}_NPC_{blaze}_blaze_{star}_star_{telluric}_telluric_{SNR}_SNR_{tell_type}_{time_dep_tell}_{wav_error}_{order_dep_throughput}_newerrun_cranked_tell_bett_airm.txt', ccfs)

