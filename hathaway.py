from scipy.interpolate import griddata as GD
from astropy.coordinates import SkyCoord
from pyshtools import legendre as pleg
from sunpy.coordinates import frames
from scipy.integrate import simps
import matplotlib.pyplot as plt
from math import sqrt, pi, e
from astropy.io import fits
import astropy.units as u
import numpy as np
import sunpy.map
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on cluster", action="store_true")
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

def get_pleg_index(l, m):
    """
    Gets the index for accessing legendre polynomials (generated from pyshtools.legendre)
    =============================================================
    Inputs: (l, m)
    =============================================================
    Outputs: index
    =============================================================
    """
    return int(l*(l+1)/2 + m)

def gen_leg(lmax, theta):
    """
    Generates associated legendre polynomials and derivatives
    
    =============================================================
    Inputs: (lmax, theta)
    -------------------------------------------------------------
    lmax      - Maximum spherical harmonic degree
    theta     - 1D array containing theta for computing P_l( cos(theta) )
    =============================================================
    Outputs: (leg, leg_d1)
    -------------------------------------------------------------
    leg       - Associated legendre polynomial
    leg_d1    - Derivative of associated legendre polynomial
    =============================================================
    """
    _cost = np.cos(theta)
    sint = np.sin(theta)
    _maxIndex = int( lmax+1 )
    _ell = np.arange(lmax+1)
    leg = np.zeros( ( _maxIndex, theta.size ) )
    leg_d1 = np.zeros( ( _maxIndex, theta.size ) )

    count = 0
    for z in cost:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2), leg_d1 * (-sint).reshape(1, sint.shape[0])/np.sqrt(2)

def gen_leg_x(lmax, x):
    """
    Generates associated legendre polynomials and derivatives
    
    =============================================================
    Inputs: (lmax, x)
    -------------------------------------------------------------
    lmax      - Maximum spherical harmonic degree
    x         - 1D array containing cos( theta )
    =============================================================
    Outputs: (leg, leg_d1)
    -------------------------------------------------------------
    leg       - Associated legendre polynomial
    leg_d1    - Derivative of associated legendre polynomial
    =============================================================
    """
    maxIndex = int( lmax+1 )
    ell = np.arange(lmax+1)
    leg = np.zeros( ( maxIndex, x.size ) )
    leg_d1 = np.zeros( ( maxIndex, x.size ) )

    count = 0
    for z in x:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2), leg_d1/np.sqrt(2)

def smooth(img):
    avg_img =(    img[1:-1 ,1:-1]  # center
                + img[ :-2 ,1:-1]  # top
                + img[2:   ,1:-1]  # bottom
                + img[1:-1 , :-2]  # left
                + img[1:-1 ,2:  ]  # right
                ) / 5.0
    return avg_img

def downsample(img, N):
    for i in range(N):
        img = smooth(img)
    return img