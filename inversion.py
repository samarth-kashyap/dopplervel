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
    cost = np.cos(theta)
    sint = np.sin(theta)
    maxIndex = int( (lmax+1)*(lmax+2)/2 )
    leg = np.zeros( ( maxIndex, theta.size ) )
    leg_d1 = np.zeros( ( maxIndex, theta.size ) )
    dp_leg = np.zeros( (maxIndex, theta.size), dtype=complex)

    count = 0
    for z in cost:
        leg[:, count], leg_d1[:, count] = pleg.PlmBar_d1(lmax, z, 1, 1)
        count += 1
    dt_leg = leg_d1 * (-sint).reshape(1, sint.shape[0])
    
    ellArr = np.zeros(maxIndex)
    emmArr = np.zeros(maxIndex)
    countm, countl = 0, 0
    for i in range(maxIndex):
        ellArr[i] = countl
        emmArr[i] = countm
        if countm==countl:
            countl += 1
            countm = 0
        else:
            countm += 1
    norm = np.sqrt( ellArr * (ellArr + 1) )
    norm[norm<1] = 1.0
    
    legtemp = leg.copy()
    dp_leg = 1j * emmArr.reshape(maxIndex, 1) * legtemp / sint.reshape(1, sint.shape[0])
    dt_leg /= norm.reshape(maxIndex, 1)
    dp_leg /= norm.reshape(maxIndex, 1)
    return leg/sqrt(2)/sqrt(2*pi), dt_leg/sqrt(2)/sqrt(2*pi), dp_leg/sqrt(2)/sqrt(2*pi)

def gen_full_mat3(t, lmax, theta):
    sint = np.sin(theta)
    cost = np.cos(theta)
    
    lr = cost; lt = -sint;
    leg, dt_leg, dp_leg = gen_leg(lmax, theta)
    
    matsize = lmax + 1 - t
    fullMat = np.zeros((matsize*3, matsize*3), dtype=complex);
    
    for i in range(matsize*3):
        ui, ess = divmod(i, matsize)
        ess += t
        for j in range(matsize*3):
            uj, ell = divmod(j, matsize)
            ell += t
            normell = np.sqrt( ell * (ell + 1) ) if ell>0 else 1.0
            if ui==0 and uj==0:
                fullMat[i, j] = simps( sint * lr**2 * leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :].conjugate(), x=theta) 
            elif ui==0 and uj==1:
                fullMat[i, j] = simps( sint * lr*lt * leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :].conjugate(), x=theta) 
            elif ui==0 and uj==2:
                fullMat[i, j] = simps( sint * lr*lt * leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :].conjugate(), x=theta)
            elif ui==1 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt * dt_leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :].conjugate(), x=theta) 
            elif ui==1 and uj==1:
                fullMat[i, j] = simps( sint * lt**2 * dt_leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :].conjugate(), x=theta) 
            elif ui==1 and uj==2:
                fullMat[i, j] = simps( sint * lt**2 * dt_leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :].conjugate(), x=theta)
            elif ui==2 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt * dp_leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :].conjugate(), x=theta)
            elif ui==2 and uj==1:
                fullMat[i, j] = simps( sint * lt**2 * dp_leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :].conjugate(), x=theta)
            elif ui==2 and uj==2:
                fullMat[i, j] = simps( sint * lt**2 * dp_leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :].conjugate(), x=theta)
    return fullMat*2*pi

def inv_SVD(A, svdlim):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    sinv = s**-1
#    '''
    plt.figure()
    plt.semilogy(sinv, '.')
    plt.axhline(y=svdlim)
    plt.title("Singular values")
    plt.show()
#    '''
    sinv[sinv/sinv[0] > svdlim] = 0.0#svdlim
    return np.dot( v.transpose().conjugate(), np.dot(np.diag(sinv), u.transpose().conjugate()))

def inv_reg1(A, regparam):
    Ashape = A.shape[0];
    return np.linalg.inv(A.transpose().conjugate().dot(A) + regparam * np.identity(Ashape)).dot(A.transpose().conjugate())

def get_only_t(lmax, t, alm, ellArr, emmArr):
    ist = emmArr==t
    isel = ( (ellArr>=t) * (ellArr<=lmax) )
    mask = ist * isel
    return alm[mask]

def put_only_t(lmax, t, alm, almFull, ellArr, emmArr):
    ist = emmArr==t
    isel = ( (ellArr>=t) * (ellArr<=lmax) )
    mask = ist * isel
    almFull[mask] = alm
    return almFull

def deconcat(alm):
    totsize = alm.shape[0]
    deconsize = int(totsize/3)
    alm1 = alm[:deconsize]
    alm2 = alm[deconsize:2*deconsize]
    alm3 = alm[2*deconsize:]
    return alm1, alm2, alm3

def computePS(alm, lmax, ellArr, emmArr):
    ps = np.zeros(lmax)
    for i in range(lmax):
        isel = ellArr==i
        ps[i] += (abs(alm[isel])**2).sum() / (2*i + 1)
    return ps

if __name__=="__main__":
    print("loading files --")
    ellArr = np.load("ellArr.txt.npz")['ellArr']
    emmArr = np.load("emmArr.txt.npz")['emmArr']
    ulmo = np.load("ulm.txt.npz")['ulm']
    vlmo = np.load("vlm.txt.npz")['vlm']
    wlmo = np.load("wlm.txt.npz")['wlm']
    print("loading files -- complete")

    ulmA = np.zeros(ulmo.shape[0], dtype=complex)
    vlmA = np.zeros(ulmo.shape[0], dtype=complex)
    wlmA = np.zeros(ulmo.shape[0], dtype=complex)

    lmaxData = int(ellArr.max())
    lmaxCalc = 200

    ell = np.arange(lmaxCalc)
    thSize = 400
    theta = np.linspace(1e-5, pi/2 - 1e-5, thSize)

    print(f"lmax = {lmaxCalc}, thSize = {thSize}")

    print("computing for .... ")
    t0 = time.time()

    for t in range(lmaxCalc):
        t1 = time.time()
        A = gen_full_mat3(t, lmaxCalc, theta)
        Ainv = inv_reg1(A, 1e-3)

        ulmt = get_only_t(lmaxCalc, t, ulmo, ellArr, emmArr)
        vlmt = get_only_t(lmaxCalc, t, vlmo, ellArr, emmArr)
        wlmt = get_only_t(lmaxCalc, t, wlmo, ellArr, emmArr)

        uot = np.concatenate((ulmt, vlmt, wlmt), axis=0)
        uAt = Ainv.dot(uot)
        uA = deconcat(uAt)

        ulmA = put_only_t(lmaxCalc, t, uA[0], ulmA, ellArr, emmArr)
        vlmA = put_only_t(lmaxCalc, t, uA[1], vlmA, ellArr, emmArr)
        wlmA = put_only_t(lmaxCalc, t, uA[2], wlmA, ellArr, emmArr)
        
        t2 = time.time()
        print(f"Time taken for t = {t}: {(t2-t1)/60} minutes")
    tn = time.time()
    print(f"Total time taken = {(tn-t0)/60} minutes")

    psu = computePS(ulmA, lmaxCalc, ellArr, emmArr)
    psv = computePS(vlmA, lmaxCalc, ellArr, emmArr)
    psw = computePS(wlmA, lmaxCalc, ellArr, emmArr)
    pstot = np.sqrt( psu**2 + psv**2 + psw**2 )

    plt.figure()
    plt.loglog(np.sqrt( ell * psu), 'g')
    plt.loglog(np.sqrt( ell * psv), 'r')
    plt.loglog(np.sqrt( ell * psw), 'b')
    plt.loglog(np.sqrt( ell * pstot), 'black')
    plt.show()

    np.savetxt("psu.txt", psu)
    np.savetxt("psv.txt", psv)
    np.savetxt("psw.txt", psw)