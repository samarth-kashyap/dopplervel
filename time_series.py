from pyshtools import legendre as pleg     # Legendre polynomials
from scipy.integrate import simps          # Integration - simpsons
import matplotlib.pyplot as plt            # Plotting
from math import sqrt, pi, e               # Math constants
import healpy as hp
import numpy as np
import argparse
import time
import os

def load_time_series(NTIME, data_dir):
    '''Loads time series of alms by reading alms of each image.

    Parameters:
    -----------
    NTIME - int 
        length of the time series 
    data_dir - string
        directory in which the alms of the image is located

    Returns:
    --------
    utime = np.ndarray(ndim=2, dtype=complex)
        time series of radial spherical harmonic
    vtime = np.ndarray(ndim=2, dtype=complex)
        time series of poloidal spherical harmonic
    wtime = np.ndarray(ndim=2, dtype=complex)
        time series of toroidal spherical harmonic

    Notes:
    ------
    axis=0 - spherical harmonic indices
    axis=1 - time

    '''
    count = 0
    for i in range(NTIME):
        suffix = str(i).zfill(3) + ".npz"
        fname = data_dir + "alm.data.inv.final" + suffix
        try:
            alm = np.load(fname)
            alm_found = True
        except FileNotFoundError:
            alm_found = False
            pass
        if count==0 and alm_found:
            ulm = alm['ulm']
            vlm = alm['vlm']
            wlm = alm['wlm']
            lmax = alm['ellmax']

            ulen = len(ulm)
            utime = np.zeros((ulen, NTIME), dtype=complex)
            vtime = np.zeros((ulen, NTIME), dtype=complex)
            wtime = np.zeros((ulen, NTIME), dtype=complex)
            utime[:, 0] = ulm
            vtime[:, 0] = vlm
            wtime[:, 0] = wlm
            print(f" i = {i} ")
            count += 1
        elif count>0 and alm_found:
            utime[:, i] = alm['ulm']
            vtime[:, i] = alm['vlm']
            wtime[:, i] = alm['wlm']
            if i%50==0:
                print(f" i = {i} ")
    return utime, vtime, wtime

def get_s_plot(ufreq, vfreq, wfreq, ellArr, emmArr, s):
    '''Takes the spectral coefficients and returns an array for a given s

    Parameters:
    -----------
    ufreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic
    vfreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic
    wfreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic
    ellArr - np.ndarray(ndim=1, dtype=int)
        list of ell values 
    emmArr - np.ndarray(ndim=1, dtype=int)
        list of emm values 
    s - int 
        spherical harmonic degree 
    t - int 
        azimuthal order 

    Returns:
    --------
    ufreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic for s, t
    vfreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic for s, t
    wfreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic for s, t

    '''
    len_alm = ufreq.shape[0]
    lmax = hp.sphtfunc.Alm.getlmax(len_alm)
    assert s<=lmax and t<=s
    isess = ellArr==s
    istee = emmArr==t
    mask = isess * istee
    ufreq_st = ufreq[mask, :]
    vfreq_st = vfreq[mask, :]
    wfreq_st = wfreq[mask, :]

    return ufreq_st, vfreq_st, w_freq_st
 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpc', help="Run program on daahpc",
                        action="store_true")
    parser.add_argument('--cchpc', help="Run program on cchpc19",
                        action="store_true")
    args = parser.parse_args()

    if args.hpc:
        home_dir = "/home/samarth/"
        scratch_dir = "/scratch/samarth/"
    if args.cchpc:
        home_dir = "/home/g.samarth/"
        scratch_dir = "/scratch/g.samarth/"
    else:
        home_dir = "/home/samarthgk/cchpchome/"
        scratch_dir = "/home/samarthgk/cchpcscratch/"

    data_dir = scratch_dir + "HMIDATA/data_analysis/"

    NTIME = 365
    max_time = NTIME*24*3600 # total time in seconds
    time_arr = np.linspace(0, max_time, NTIME)
    dtime = time_arr[1] - time_arr[0]
    freq_arr = np.fft.fftfreq(time_arr.shape[0], dtime)
    dfreq = freq_arr[1] - freq_arr[0]

    utime, vtime, wtime = load_time_series(NTIME, data_dir)
    ufreq = np.fft.fft(utime, axis=1, norm="ortho")
    vfreq = np.fft.fft(vtime, axis=1, norm="ortho")
    wfreq = np.fft.fft(wtime, axis=1, norm="ortho")

    length_alm = ufreq.shape[0]
    lmax = hp.sphtfunc.Alm.getlmax(length_alm)
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(lmax)

    _max = abs(ufreq).max()

    plt.figure()
    im = plt.imshow(ufreq.real, aspect=ufreq.shape[1]/ufreq.shape[0],
                    vmax=_max/100, vmin=-_max/100, cmap="seismic")
    plt.colorbar(im)
    plt.show()
