"""Generate synthetic velocity spectra by fitting Hathaway et al. (2015) data.

This module fits polynomial functions to the observed power spectra from
Hathaway et al. (2015) to create smooth synthetic spectra for testing
the inversion algorithm. It performs piecewise polynomial fits to better
capture the spectral shape.

References
----------
Hathaway, D. H., Teil, T., Norton, A. A., & Kitiashvili, I. (2015),
"The Sun's Photospheric Convection Spectrum", ApJ 811, 105
doi: 10.1088/0004-637X/811/2/105
arXiv: 1508.03022
"""

import matplotlib.pyplot as plt
from math import sqrt, pi, e
import numpy as np
import argparse
import os
"""
Warning (flycheck): Syntax checker python-pylint reported too many errors (420) and is disabled.
Use ‘M-x customize-variable RET flycheck-checker-error-threshold’ to
change the threshold or ‘C-u C-c ! x’ to re-enable the checker.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on cluster", action="store_true")
parser.add_argument('--deg', type=int, help="Maximum degree of polynomial fit")
parser.add_argument('--u', help="Fit for radial velocity", action="store_true")
parser.add_argument('--v', help="Fit for poloidal velocity", action="store_true")
parser.add_argument('--w', help="Fit for toroidal velocity", action="store_true")
args = parser.parse_args()

if args.hpc:
	homeDir = "/home/samarth/"
	scratchDir = "/scratch/samarth/"
else:
	homeDir = "/home/samarthgk/hpchome/"
	scratchDir = "/home/samarthgk/hpcscratch/"

import sys; sys.path.append( homeDir )
from heliosPy import datafuncs as cdata
from heliosPy import mathfuncs as cmath
from heliosPy import iofuncs as cio

def piecewise(pieces, Nmax):
    totInt = pieces.shape[0] - 1
    for i in range(totInt):
        pieceMin, pieceMax = pieces[i], pieces[i+1]
        z = np.polyfit(ulm[pieceMin:pieceMax, 0], ulm[pieceMin:pieceMax, 1]/np.sqrt(ulm[pieceMin:pieceMax, 0]), 7)

def get_fit_coefs(data, breaks, deg):
    """Fit input curve piecewise using polynomials.

    Parameters
    ----------
    data : np.ndarray(ndim=2)
        Data array where column 0 contains x values and column 1 contains y values
    breaks : np.ndarray(ndim=1, dtype=int)
        Array of indices indicating breaks for piecewise fitting.
        E.g., breaks = [0, 12, 56] fits two polynomials: one for indices 0-12
        and another for indices 12-56
    deg : int
        Maximum degree of fitted polynomials
    
    Returns
    -------
    pfit : list of np.ndarray
        List of polynomial coefficient arrays, one for each piece.
        Use np.poly1d(pfit[i]) to obtain the polynomial function for piece i
    
    Notes
    -----
    The function fits y/sqrt(x) vs x to better capture the spectral shape.
    """
#    pfit = np.zeros((breaks.shape[0]-1, deg+1))
    pfit = []
    for i in range(breaks.shape[0]-1):
        pieceMin = breaks[i]
        pieceMax = breaks[i+1]
        xdata = data[pieceMin:pieceMax, 0]
        ydata = data[pieceMin:pieceMax, 1]
        z = np.polyfit(xdata, ydata/np.sqrt(xdata), deg, full=True)
#        pfit[i, :] = z[0]
        pfit.append(z[0])
        print(f"LS error for fit (for piece = {i}): {z[1]}")
    return pfit

def get_fit_poly(data, breaks, pfit):
    polyList = []
    xList = []
    fitShape = len(pfit)
    for i in range(fitShape):
#        p = np.poly1d(pfit[i, :])
        p = np.poly1d(pfit[i])
        x = np.linspace(data[breaks[i], 0], data[breaks[i+1], 0], 1000)
        newu = p(x)
        polyList.append(newu)
        xList.append(x)
    return xList, polyList

if __name__=="__main__":
    # Loading the data points from Hathaway et al. (2015) - 
    # "The Sun's Photospheric Convection Spectrum", ApJ 811, 105
    dataDir = homeDir + "dopplervel2/"
    ulm = np.loadtxt(dataDir + "green.csv", delimiter=",")          # radial spectrum
    vlm = np.loadtxt(dataDir + "red.csv", delimiter=",")            # poloidal spectrum 
    wlm = np.loadtxt(dataDir + "blue.csv", delimiter=",")           # toroidal spectrum
    totlm = np.loadtxt(dataDir + "black.csv", delimiter=",")        # total spectrum

    if args.u:
        data = ulm.copy()
        splitPoint = 32
    elif args.v:
        data = vlm.copy()
        splitPoint = 52
    elif args.w:
        data = wlm.copy()
        splitPoint = 47
    else:
        data = ulm.copy()
        splitPoint = 32

    # maximum degree for fitting polynomial
    degMax = args.deg

    # get the polynomial coefficients
    ztemp = get_fit_coefs(data, np.array([0, splitPoint, data.shape[0]-1]), degMax)

    # obtain the polynomial as a function of ell
    xtemp, ptemp = get_fit_poly(data, np.array([0, splitPoint, data.shape[0]-1]), ztemp)

    # storing the coefficients in a file
    if args.u:
        np.savez_compressed( dataDir + "u_poly.npz", u=ztemp, ellu=xtemp )
    elif args.v:
        np.savez_compressed( dataDir + "v_poly.npz", v=ztemp, ellv=xtemp )
    elif args.w:
        np.savez_compressed( dataDir + "w_poly.npz", w=ztemp, ellw=xtemp )
    else:
        np.savez_compressed( dataDir + "u_poly.npz", u=ztemp, ellu=xtemp )

    # plotting data and fit
    xdata, ydata = data[:, 0].copy(), data[:, 1]/np.sqrt(data[:, 0])
    plt.figure()
    for i in range(len(xtemp)):
        plt.loglog(xtemp[i], ptemp[i], 'b')
    plt.loglog(xdata, ydata, '.r', alpha = 0.7)
    plt.show(); plt.close()
