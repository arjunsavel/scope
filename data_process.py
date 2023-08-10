"""
Helper functions to rework data arrays.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splrep, splev, interp1d
from scipy import interpolate
from scipy import stats
import pdb


def getSpacing(arr):
    """
    Gets the spacing between points in an array.

    Inputs
    ------
        :arr: [array] array that you want spacing for.
    """
    return (arr[-1] - arr[0]) / (len(arr) - 1)

def fwhm2sigma(fwhm):
    """
    Converts the full-width half-max to a sigma.
    """
    return fwhm / np.sqrt(8 * np.log(2))

def gaussian(x, mu, sig):
    """
    computes a gaussian given inputs x, mean of the distribution, and sigma
    """
    normalization = 1 / np.sqrt(2 * np.pi * sig**2)
    exponent = -((x - mu) ** 2 / (2 * sig**2))
    y = normalization * np.exp(exponent)
    return y

def reduceSpectralResolution(x, y, R_low, R_high=None, lambda_mid=None, n=4):
    """
    Takes the spectra resolution of a spectrum and reduces it.

    Inputs
    ------
        :x: [array] wavelength.
        :y: [array] flux, or whatever.
        :R_low: [int] resolution to convolve down to.
        :R_high: [int] current resolution. can be inferred.
        :lambda_mid: [float] middle wavelength. can be inferred.
        :n: [int]

    Outputs
    -------
        :lowres: [array] the convolved-down y.
    """
    if np.ndim(y) == 2:
        return np.array(
            [
                reduceSpectralResolution(x, spec, R_low, R_high, lambda_mid, n)
                for spec in y
            ]
        )

    dx = getSpacing(x)

    # If lambda_mid is none, take median of input wavelengths
    if lambda_mid is None:
        lambda_mid = np.median(x)

    # If R_high is none, use midpoint of x divided by spacing in x
    if R_high is None:
        R_high = lambda_mid / dx

    # Create Gaussian kernel
    fwhm = np.sqrt(R_high**2 - R_low**2) * (lambda_mid / (R_low * R_high))
    sigma = fwhm2sigma(fwhm)

    kernel_x = np.arange(-n * sigma, n * sigma + dx, dx)
    kernel = gaussian(kernel_x, 0, sigma)
    kernel = kernel / np.sum(kernel)

    # find center of kernel
    n_kernel_lt0 = len(np.where(kernel_x < 0)[0])
    n_kernel_gt0 = len(np.where(kernel_x > 0)[0])

    if n_kernel_lt0 < n_kernel_gt0:
        origin = 0
    else:
        origin = -1

    # convolve
    lowRes = ndi.convolve(y, kernel, origin=origin)
    return lowRes




def initialize_knots(wmin, wmax, knot_spacing):
    """Place knots evenly through"""
    waverange = wmax - wmin - 2 * knot_spacing
    Nknots = int(waverange // knot_spacing)
    minknot = wmin + (waverange - Nknots * knot_spacing) / 2.0
    xknots = np.arange(minknot, wmax, knot_spacing)
    # Make sure there the knots don't hit the edges
    while xknots[-1] >= wmax - knot_spacing:
        xknots = xknots[:-1]
    while xknots[0] <= wmin + knot_spacing:
        xknots = xknots[1:]
    return list(xknots)

def fit_continuum_lsq(
    spec,
    knots,
    exclude=[],
    maxiter=3,
    sigma_lo=2,
    sigma_hi=2,
    get_edges=False,
    **kwargs,
):
    """Fit least squares continuum through spectrum data using specified knots, return model"""
    assert np.all(np.array(list(map(len, exclude))) == 2), exclude
    assert np.all(np.array(list(map(lambda x: x[0] < x[1], exclude)))), exclude
    x, y, w = (
        np.linspace(0, len(spec), len(spec)),
        np.array(spec),
        np.array([x / 10 for x in np.array(spec)]),
    )

    # This is a mask marking good pixels
    mask = np.ones_like(x, dtype=bool)
    # Exclude regions
    for xmin, xmax in exclude:
        mask[(x >= xmin) & (x <= xmax)] = False
    # Get rid of bad fluxes
    mask[np.abs(y) < 1e-6] = False
    mask[np.isnan(y)] = False
    if get_edges:
        left = np.where(mask)[0][0]
        right = np.where(mask)[0][-1]

    for iter in range(maxiter):
        # Make sure there the knots don't hit the edges
        wmin = x[mask].min()
        wmax = x[mask].max()
        while knots[-1] >= wmax:
            knots = knots[:-1]
        while knots[0] <= wmin:
            knots = knots[1:]

        try:
            fcont = interpolate.LSQUnivariateSpline(
                x[mask], y[mask], knots, w=w[mask], **kwargs
            )
        except ValueError:
            print("Knots:", knots)
            print("xmin, xmax = {:.4f}, {:.4f}".format(wmin, wmax))
            raise
        # Iterative rejection
        cont = fcont(x)
        sig = (cont - y) * np.sqrt(w)
        sig /= np.nanstd(sig)
        mask[sig > sigma_hi] = False
        mask[sig < -sigma_lo] = False
    if get_edges:
        return fcont, left, right
    return fcont

def spork(spec, N_knots, x, sigma_hi, sigma_low):
    return fit_continuum_lsq(
        np.array(spec),
        knots=np.array(initialize_knots(0, len(x), N_knots)).astype(int),
        maxiter=5,
        sigma_lo=sigma_low,
        sigma_hi=sigma_hi,
        get_edges=False,
    )(np.linspace(0, len(x), len(x)))


""" More aggressive masking based on the number of standard deviations of a 
column above the average standard deviation. Takes in input the telluric
removed, prossibly already masked matrix. """

def mask_data(spec):
    """
    seems to be something that like catches bad spline evaluations?
    """
    # todo: change this to whatever it'll look like later on.
    no, nf, nx = spec.shape
    # no = 4 detectors
    # nf = 59 exposures
    # nx = 1024 pixels
    for io in range(no):
        spc = spec[
            io,
        ].copy()
        sig = np.std(spc, axis=0)
        mstd = np.median(sig)
        for i in range(nx):
            if sig[i] > 2.0 * mstd:
                spc[:, i] = 0.0
        spec[
            io,
        ] = spc
    return spec



def get_fixed_ccf_cube(spec, wl, rv, cs):
    no, nf, nx = spec.shape
    ncc = len(rv)
    tr = np.zeros((no, nf, ncc))
    idMat = np.ones(nx)
    N = float(nx)
    for ic in range(ncc):
        beta = -rv[ic] / 2.998e5
        wShift = wl * np.sqrt((1 + beta) / (1 - beta))
        fShift = splev(wShift, cs, der=0)
        for io in range(no):
            gVec = fShift[
                io,
            ].copy()
            gVec -= (gVec @ idMat) / N
            sg2 = (gVec @ gVec) / N
            pdb.set_trace()
            for j in range(nf):
                fVec = spec[
                    io,
                    j,
                ].copy()
                fVec -= (fVec @ idMat) / N
                sf2 = (fVec @ fVec) / N
                R = (fVec @ gVec) / N
                tr[io, j, ic] = R / np.sqrt(sf2 * sg2)
    return tr

def get_velocity_map(ccf, vIn, ph, vBary, vSys, vRest, kpVec):
    nf, ncc = ccf.shape
    nkp = len(kpVec)
    nvs = len(vRest)
    diag = np.zeros((nkp, nvs))
    trail = np.zeros((nf, nvs))

    # iterate over the Kp to create the map
    for ik in range(nkp):
        vPl = vSys + vBary + kpVec[ik] * np.sin(2.0 * np.pi * ph)
        for j in range(nf):
            try:
                xin = vIn - vPl[j]
                fit = interp1d(
                    xin,
                    ccf[
                        j,
                    ],
                )
                trail[
                    j,
                ] = fit(vRest)
            except:
                pass

        diag[
            ik,
        ] = np.mean(trail, axis=0)
    test = diag.flatten()
    test = np.sort(test)
    sig = np.mean([-test[int(nvs * nkp * 0.34)], test[int(nvs * nkp * 0.67)]])
    diag /= sig
    return diag, vRest, kpVec

def do_ttest(din, dout):
    # Getting dimensions
    din = din.flatten()
    dout = dout.flatten()
    nin = len(din)
    nout = len(dout)

    # Computing means and variances
    mOut = np.mean(dout)
    mIn = np.mean(din)
    sOut = ((dout[:] - mOut) ** 2.0).sum() / (nout - 1.0)
    sIn = ((din[:] - mIn) ** 2.0).sum() / (nin - 1.0)

    # Computing the t statistic and degrees of freedom
    t = (mIn - mOut) / np.sqrt(sOut / nout + sIn / nin)
    nu = (sOut / nout + sIn / nin) ** 2.0 / (
        ((sOut / nout) ** 2.0 / (nout - 1)) + ((sIn / nin) ** 2.0 / (nin - 1))
    )

    p2t = stats.t.sf(t, nu)

    return p2t, stats.norm.isf(p2t)  # /2.0)

def get_ttest_map(ccf, vIn, vOut, rvVec, phVec, vLag, vRest, kpVec):
    nf, ncc = ccf.shape
    nk = len(kpVec)
    nv = len(vRest)
    ttMap = np.zeros((nk, nv))
    for ik in range(nk):
        for iv in range(nv):
            rvDist = np.zeros((nf, ncc))
            rvPl = rvVec + kpVec[ik] * np.sin(2.0 * np.pi * phVec) + vRest[iv]
            for j in range(nf):
                rvDist[
                    j,
                ] = np.abs(vLag - rvPl[j])
            inTrail = ccf[rvDist < vIn]
            outTrail = ccf[rvDist > vOut]
            p2t, sigma = do_ttest(inTrail, outTrail)
            ttMap[ik, iv] = sigma
    return ttMap

def get_likelihood(spec, wl, rv, cs):
    """

    I need to add this up for each exposure, right?
    """
    no, nf, nx = spec.shape
    ncc = len(rv)
    tr = np.zeros((no, nf, ncc))
    idMat = np.ones(nx)
    N = float(nx)
    for ic in range(ncc):
        beta = -rv[ic] / 2.998e5
        wShift = wl * np.sqrt((1 + beta) / (1 - beta))
        fShift = splev(wShift, cs, der=0)
        for io in range(no):
            gVec = fShift[
                io,
            ].copy()
            gVec -= (gVec @ idMat) / N
            sg2 = (gVec @ gVec) / N
            for j in range(nf):
                fVec = spec[
                    io,
                    j,
                ].copy()
                fVec -= (fVec @ idMat) / N
                sf2 = (fVec @ fVec) / N
                R = (fVec @ gVec) / N
                tr[io, j, ic] = (-0.5*N * np.log(sf2 + sg2 - 2.0*R))
    return tr

def get_likelihood_map(like_cube):
    """
    takes a likelihood cube and turns it into a map.
    """
    no, nf, ncc = like_cube.shape
    for j in range(nf):

    # sum across exposures

    return like_cube - like_cube.min()