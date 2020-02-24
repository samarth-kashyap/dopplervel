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
    for z in _cost:
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

def inv_SVD(A, svdlim):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    sinv = s**-1
    '''
    plt.figure()
    plt.semilogy(sinv, '.')
    plt.axhline(y=svdlim)
    plt.title("Singular values")
    plt.show()
    '''
    sinv[sinv/sinv[0] > svdlim] = 0.0#svdlim
    return np.dot( v.transpose().conjugate(), np.dot(np.diag(sinv), u.transpose().conjugate()))

def inv_reg1(A, regparam):
    Ashape = A.shape[0];
    return np.linalg.inv(A.transpose().conjugate().dot(A) + regparam * np.identity(Ashape)).dot(A.transpose().conjugate())

def inv_reg2(A, regparam):
    reg2 = 2*np.identity(A.shape[0])
    offd2 = -1*np.identity(A.shape[0]-1)
    reg2[1:, :-1] += offd2
    reg2[:-1, 1:] += offd2
    reg = reg2[1:-1, :].copy()
    return np.linalg.inv(A.transpose().dot(A) + (regparam/16.) * reg.transpose().dot(reg)).dot(A.transpose())

def inv_reg3(A, regparam):
    reg2 = 3*np.identity(A.shape[0])
    offd2 = -1*np.identity(A.shape[0]-1)
    reg2[:-1, 1:] += 3*offd2
    reg2[1:, :-1] += offd2
    reg2[:-2, 2:] += -offd2[1:, 1:]
    reg = reg2[1:-2, :].copy()
    return np.linalg.inv(A.transpose().dot(A) + (regparam/64.) * reg.transpose().dot(reg)).dot(A.transpose())


if __name__ == "__main__":
    hmiDataDir = scratchDir + "HMIDATA/v720s_dConS/2018/"
    hmiDataFileNames = hmiDataDir + "HMI_2018_filenames"
    plotDir = scratchDir + "plots/dopplervel/"

    with open(hmiDataFileNames, mode="r") as f:
        hmiFiles = f.read().splitlines()

    totalDays = len(hmiFiles)
    print(f"Total days = {totalDays}")

    try:
        procid=int(os.environ['PBS_VNODENUM'])
    except KeyError: pass
    nproc = 12 
    daylist = np.arange(procid, totalDays, nproc)
    print(f" procid = {procid}")

    for day in daylist:#range(totalDays):#daylist:
        # loading HMI image as sunPy map
        hmi_map = sunpy.map.Map(hmiDataDir + hmiFiles[day])

        # Getting the B0 and P0 angles
        B0 = hmi_map.observer_coordinate.lat
        P0 = hmi_map.observer_coordinate.lon

        # setting up for data cleaning
        x, y = np.meshgrid(*[np.arange(v.value) for v in hmi_map.dimensions]) * u.pix
        hpc_coords = hmi_map.pixel_to_world(x, y)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / hmi_map.rsun_obs

        # removing all data beyond rcrop (heliocentric radius)
        rcrop = 0.95
        maskR = r>rcrop

        hmi_map.data[maskR] = np.nan
        r[maskR] = np.nan

        hpc_hgf = hpc_coords.transform_to(frames.HeliographicStonyhurst)
        hpc_hc = hpc_coords.transform_to(frames.Heliocentric)

        mapData = hmi_map.data.copy()
        maskNaN = ~np.isnan(mapData)

        # -- ( plot ) determining maximum and minimum pixel values --
        maxData = mapData[maskNaN].max()
        minData = mapData[maskNaN].min()
        maxx = max(abs(maxData), abs(minData))
        print(maxData, minData)
        del mapData, maxData, minData
        # ------------------------------------------------------

        """
        plt.figure()
        im = plt.imshow(hmi_map.data, cmap='seismic', interpolation="none")
        plt.colorbar(im)
        plt.axis("off")
        plt.savefig(plotDir + "rawMap.png", dpi=500)
        plt.show()
        """

        # Removing effect of satellite velocity
        mapData = hmi_map.data.copy()
        maskNaN = ~np.isnan(mapData)

        mapFits = fits.open(hmiDataDir + hmiFiles[0])
        mapFits.verify('fix')
        vx = mapFits[1].header['OBS_VR']
        vz = mapFits[1].header['OBS_VN']
        vy = mapFits[1].header['OBS_VW']
        print(f"VR = {vx}, VN = {vz}, VW = {vy}")

        VR, VN, VW = vx, vz, vy

        x = hpc_hc.x
        y = hpc_hc.y
        sChi = np.sqrt( x**2 + y**2 )
        sSig = np.sin(np.arctan(r/200))
        cSig = np.cos(np.arctan(r/200))
        chi = np.arctan2(x, y)
        sChi = np.sin(chi)
        cChi = np.cos(chi)

        thetaHGC = hpc_hgf.lat.copy()# + 90*u.deg
        phiHGC = hpc_hgf.lon.copy()
        thHG1 = thetaHGC[maskNaN]
        phHG1 = phiHGC[maskNaN]
        del thetaHGC, phiHGC

        ct = np.cos(thHG1)
        st = np.sin(thHG1)
        cp = np.cos(phHG1)
        sp = np.sin(phHG1)

        vr1 = cp*st*vx + sp*st*vy + ct*vz
        vt1 = cp*ct*vx + sp*ct*vy - st*vz
        vp1 = -sp*vx + cp*vy 

        sB0 = np.sin(B0)
        cB0 = np.cos(B0)

        lr = sB0*ct + cB0*st*cp
        lt = sB0*st - cB0*ct*cp
        lp = cB0*sp

        vC1 = lr*vr1 + lt*vt1 + lp*vp1
        velCorr = np.zeros((4096, 4096))
        velCorr[maskNaN] = vC1
        velCorr[~maskNaN] = np.nan

        # method 2
        vr1 = VR*cSig
        vr2 = -VW*sSig*sChi
        vr3 = -VN*sSig*cChi
        vC1 = vr1 + vr2 + vr3
        velCorr = np.zeros((4096, 4096))
        velCorr[maskNaN] = vC1[maskNaN] + 632
        velCorr[~maskNaN] = np.nan

        tempFull = np.zeros((4096, 4096))
        tempFull[~maskNaN] = np.nan

        mapData += velCorr 
        mapData -= 632

        lat = hpc_hgf.lat.copy() + 90*u.deg
        lon = hpc_hgf.lon.copy()
        cB0 = np.cos(B0)
        sB0 = np.sin(B0)

        lat1D = lat[maskNaN].copy()
        lon1D = lon[maskNaN].copy()
        rho1D = r[maskNaN].copy()
        ct = np.cos(lat1D)
        st = np.sin(lat1D)
        sp = np.sin(lon1D)
        cp = np.cos(lon1D)

        t1 = time.time()
        pl_theta, dt_pl_theta = gen_leg(5, lat1D)
        ell = np.arange(6)
        t2 = time.time()
        print(f"Time taken for pl_theta = {(t2 - t1)/60} minutes")
        pl_rho, dt_pl_rho = gen_leg_x(5, rho1D)
        t3 = time.time()
        print(f"Time taken for pl_rho = {(t3 - t2)/60} minutes")

        # Normalizing the functions
        dt_pl_theta /= np.sqrt(ell*(ell+1)).reshape(6, 1)
        dt_pl_rho /= np.sqrt(ell*(ell+1)).reshape(6, 1)

        lt = sB0 * st - cB0 * ct * cp
        lp = cB0 * sp
        imArr = np.zeros((11, lt.shape[0]))
        A = np.zeros((11, 11))
        imArr[0, :] = dt_pl_theta[1, :] * lp
        imArr[1, :] = dt_pl_theta[3, :] * lp
        imArr[2, :] = dt_pl_theta[5, :] * lp

        #imArr[3, :] = dt_pl_theta[0, :] * lt
        imArr[3, :] = dt_pl_theta[2, :] * lt
        imArr[4, :] = dt_pl_theta[4, :] * lt

        imArr[5, :] = pl_rho[0, :]
        imArr[6, :] = pl_rho[1, :]
        imArr[7, :] = pl_rho[2, :]
        imArr[8, :] = pl_rho[3, :]
        imArr[9, :] = pl_rho[4, :]
        imArr[10, :] = pl_rho[5, :]

        mapArr = mapData[maskNaN].copy()

        RHS = imArr.dot(mapArr)

        for i in range(11):
            for j in range(11):
                A[i, j] = imArr[i, :].dot(imArr[j, :])

        Ainv = inv_SVD(A, 1e5)
        fitParams = Ainv.dot(RHS)
        print(f" ##(in m/s)## Rotation = {fitParams[:3]}, \nMeridional Circ = {fitParams[3:5]},\nLimb Shift = {fitParams[5:]}\n")
        print(f" ##(in Hz)## Rotation = {fitParams[:3]/2/pi/0.695}, \nMeridional Circ = {fitParams[3:5]/2/pi/0.695},\nLimb Shift = {fitParams[5:]}")
        newImgArr = fitParams.dot(imArr)
        new1 = fitParams[:3].dot(imArr[:3, :])
        new2 = fitParams[3:5].dot(imArr[3:5, :])
        new3 = fitParams[5:].dot(imArr[5:, :])

        newImg = np.zeros((4096, 4096))
        newImg[maskNaN] = newImgArr
        newImg[~maskNaN] = np.nan
        resImg = mapData - newImg

        cio.writefitsfile(mapData - newImg, hmiDataDir + "residual"+str(day).zfill(3)+".fits")
        cio.writefitsfile((hpc_hgf.lat + 90*u.deg).value, hmiDataDir + "theta"+str(day).zfill(3)+".fits")
        cio.writefitsfile(hpc_hgf.lon.value , hmiDataDir + "phi"+str(day).zfill(3)+".fits")
        del mapData, newImg, hpc_hgf, resImg, new1, new2, new3, newImgArr, fitParams, Ainv, imArr
        del dt_pl_rho, dt_pl_theta, st, ct, sp, cp
#        cio.writefitsfile(theta.value, hmiDataDir + "thetaRot"+str(i).zfill(3)+".fits")
#        cio.writefitsfile(phi.value, hmiDataDir + "phiRot"+str(i).zfill(3)+".fits")
