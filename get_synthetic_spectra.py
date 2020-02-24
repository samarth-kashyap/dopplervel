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
    """
    Fit the input curve piecewise using polynomials.

    Inputs: (data, breaks, deg)
    ----------------------------------------------------
    data    (np.ndarray(ndim=2)) - dim=1 has data of x, 
                                   dim=2 has data of y
    breaks  (np.ndarray(ndim=1)) - array of indices to indicate breaks (for piecewise fitting)
                                   E.g. breaks = np.ndarray([0, 12, 56]) then 2 different polynomials
                                   are fitted, one for (0, 12) and another for (12, 56)
    deg     (        int       ) - maximum degree of fitted polynomials
    
    Outputs: (pfit)
    ----------------------------------------------------
    pfit    (list) - np.ndarray for each piece
    E.g. to obtain the curve of second piece, use np.poly1d(pfit[1])
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
    # loading the data points from Hathaway 2015.
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
