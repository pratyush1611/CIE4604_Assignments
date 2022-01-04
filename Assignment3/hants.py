from copy import deepcopy
#import pandas as pd
import math
import numpy as np

def hants(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta):
    """Harmonic ANalysis of Time Series (HANTS)
    
    Parameters
    ----------
    ni :  int
        Number of images (total number of actual samples of the time series)
    nb : int 
        Length of the base period, measured in virtual samples (days, dekads, months, etc.)
    nf: int
        Number of frequencies to be considered above the zero frequency
    y: ndarray
        1D array of size size `ni` with input sample values (e.g. NDVI values)
    ts: ndarray
        1D array of size `ni` of time sample indicators (indicates virtual 
        sample number relative to the base period); numbers in array `ts` 
        maybe greater than `nb`
    HiLo: str {'Hi', 'Lo'}
        2-character string indicating rejection of high 'Hi' or low 'Lo' outliers
    low: float
        valid range minimum
    high: float
        valid range maximum (values outside the valid range are rejeced right away)
    fet: float
        fit error tolerance (points deviating more than fet from curve fit are rejected)
    dod : int
        degree of overdeterminedness (iteration stops if number of points reaches 
        the minimum required for curve fitting, plus dod). This is a safety measure
    delta : float
        small positive number (e.g. 0.1) to suppress high amplitudes

    Returns
    -------
    amp : ndarray
        1D array of length `nf`+` with Amplitude for frequencies 0...nf
    phi : ndarray
        1D array of length `nf'+1 with Phase in degrees for frequencies 0...nf
    yr : ndarray
        1D array of length `ni` with smoothed values (e.g. NDVI)   
    outliers : ndarray 
        1D Boolean array of length `ni` with flagged outliers
     

    Notes
    -----    
    This function applies the Harmonic ANalysis of Time Series (HANTS)
    algorithm originally developed by the Netherlands Aerospace Centre (NLR)
    (http://www.nlr.org/space/earth-observation/).

    This python implementation was based on two previous implementations
    available at the following links:
    https://codereview.stackexchange.com/questions/71489/harmonic-analysis-of-time-series-applied-to-arrays
    http://nl.mathworks.com/matlabcentral/fileexchange/38841-matlab-implementation-of-harmonic-analysis-of-time-series--hants-

    Original Author for Python Version: Espinoza-Dávalos, G. E., Bastiaanssen, W. G. M., Bett, B., & Cai, X. (2017).
    A Python Implementation of the Harmonic ANalysis of Time Series (HANTS) Algorithm for Geospatial Data.
    http://doi.org/10.5281/zenodo.820623

    Edited by: Ullas Rajvanshi and Hans van der Marel
    """
    
    # Arrays
    mat = np.zeros((min(2 * nf + 1, ni), ni))
    amp = np.zeros((nf + 1))

    phi = np.zeros((nf + 1))
    yr = np.zeros((ni, 1))
    #y_len = len(y)
    #outliers = np.zeros((1, y_len))
    outliers = np.zeros((ni, 1))

    # Filter
    sHiLo = 0
    if HiLo == 'Hi':
        sHiLo = -1
    elif HiLo == 'Lo':
        sHiLo = 1

    nr = min(2 * nf + 1, ni)
    noutmax = ni - nr - dod
    dg = 180.0 / math.pi
    mat[0, :] = 1.0

    ang = 2 * np.pi * np.arange(nb) / nb
    cs = np.cos(ang)
    sn = np.sin(ang)

    i = np.arange(1, nf + 1)
    for j in np.arange(ni):
        index = np.mod(i * ts[j], nb)
        mat[2 * i - 1, j] = cs.take(index)
        mat[2 * i, j] = sn.take(index)

    p = np.ones_like(y)
    bool_out = (y < low) | (y > high)
    p[bool_out] = 0
    #outliers[bool_out.reshape(1, y.shape[0])] = 1
    outliers[bool_out] = 1
    nout = np.sum(p == 0)
    # ready state set to be false
    # if nout > noutmax:
    #     if pd.np.isclose(y, fill_val).any():
    #         ready = pd.np.array([True])
    #         yr = y
    #         outliers = pd.np.zeros((y.shape[0]), dtype=int)
    #         outliers[:] = fill_val
    #     else:
    #         raise Exception('Not enough data points.')
    # else:
    #     ready = pd.np.zeros((y.shape[0]), dtype=bool)

    nloop = 0
    ready = False
    nloopmax = ni

    while (not ready) & (nloop < nloopmax):

        nloop += 1
        za = np.matmul(mat, p * y)

        A = np.matmul(np.matmul(mat, np.diag(p)),
                         np.transpose(mat))
        A = A + np.identity(nr) * delta
        A[0, 0] = A[0, 0] - delta

        zr = np.linalg.solve(A, za)

        yr = np.matmul(np.transpose(mat), zr)
        diff_vec = sHiLo * (yr - y)
        err = p * diff_vec

        err_ls = list(err)
        err_sort = deepcopy(err)
        err_sort.sort()

        rank_vec = [err_ls.index(f) for f in err_sort]

        maxerr = diff_vec[rank_vec[-1]]
        ready = (maxerr <= fet) | (nout == noutmax)

        if not ready:
            i = ni - 1
            j = rank_vec[i]
            while (p[j] * diff_vec[j] > 0.5 * maxerr) & (nout < noutmax):
                p[j] = 0
                #outliers[0, j] = 1
                outliers[j] = 1
                nout += 1
                i -= 1
                if i == 0:
                    j = 0
                else:
                    j = 1

    amp[0] = zr[0]
    phi[0] = 0.0
    i = np.arange(2, nr, 2)
    ifr = (i + 2) // 2
    ra = zr[i-1]
    rb = zr[i]
    amp[ifr-1] = np.sqrt(ra * ra + rb * rb)
    phase = np.arctan2(rb, ra) * dg
    phase[phase < 0] = phase[phase < 0] + 360
    phi[ifr-1] = phase
    
    outliers = np.squeeze(outliers == 1)

    return amp, phi, yr, outliers

if __name__ == '__main__':
    hants()
