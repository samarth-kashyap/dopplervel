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

def gen_leg_real(lmax, theta):
    cost = np.cos(theta)
    sint = np.sin(theta)
    maxIndex = int( (lmax+1)*(lmax+2)/2 )
    leg = np.zeros( ( maxIndex, theta.size ) )
    leg_d1 = np.zeros( ( maxIndex, theta.size ) )
    dp_leg = np.zeros( (maxIndex, theta.size) )

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
    dp_leg = emmArr.reshape(maxIndex, 1) * legtemp / sint.reshape(1, sint.shape[0])
    dt_leg /= norm.reshape(maxIndex, 1)
    dp_leg /= norm.reshape(maxIndex, 1)
    return leg/sqrt(2)/sqrt(2*pi), dt_leg/sqrt(2)/sqrt(2*pi), dp_leg/sqrt(2)/sqrt(2*pi)

def vel_from_spectra_allt(ulm, vlm, wlm, thSize, phSize, lmax):
    theta = np.linspace(1e-5, pi-1e-5, thSize)
    phi = np.linspace(1e-5, 2*pi - 1e-5, phSize)
    
    leg, dt_leg, dp_leg = gen_leg(lmax, theta)
    maxIndex = int( (lmax+1)*(lmax+2)/2 )
    
    ur = np.zeros( (thSize, phSize), dtype=complex)
    ut = np.zeros( (thSize, phSize), dtype=complex)
    up = np.zeros( (thSize, phSize), dtype=complex)
    
    countt, countm, countl = 0, 0, 0
    for i in range(maxIndex):
        t = countm
        costp = np.cos(t*phi)
        sintp = np.sin(t*phi)
        eitp = (costp + 1j*sintp).reshape(1, phSize)

        ur += ulm[i] * leg[i, :].reshape(thSize, 1) * eitp

        ut += vlm[i] * dt_leg[i, :].reshape(thSize, 1) * eitp
        ut -= wlm[i] * dp_leg[i, :].reshape(thSize, 1) * eitp 

        up += vlm[i] * dp_leg[i, :].reshape(thSize, 1) * eitp
        up += wlm[i] * dt_leg[i, :].reshape(thSize, 1) * eitp 
            
        if countm==countl:
            countm = 0
            countl += 1
        else:
            countm += 1
    return ur, ut, up

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

def gen_full_mat3_real(t, lmax, theta):
    sint = np.sin(theta)
    cost = np.cos(theta)
    
    lr = cost; lt = -sint;
    leg, dt_leg, dp_leg = gen_leg_real(lmax, theta)
    
    matsize = lmax + 1 - t
    fullMat = np.zeros((matsize*3, matsize*3));
    
    for i in range(matsize*3):
        ui, ess = divmod(i, matsize)
        ess += t
        for j in range(matsize*3):
            uj, ell = divmod(j, matsize)
            ell += t
            if ui==0 and uj==0:
                fullMat[i, j] = simps( sint * lr**2 * leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==0 and uj==1:
                fullMat[i, j] = simps( sint * lr*lt * leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==0 and uj==2:
                fullMat[i, j] = -simps( sint * lr*lt * leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==1 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt * dt_leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==1 and uj==1:
                fullMat[i, j] = simps( sint * lt**2 * dt_leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==1 and uj==2:
                fullMat[i, j] = -simps( sint * lt**2 * dt_leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==2 and uj==0:
                fullMat[i, j] = -simps( sint * lr*lt * dp_leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==2 and uj==1:
                fullMat[i, j] = -simps( sint * lt**2 * dp_leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==2 and uj==2:
                fullMat[i, j] = simps( sint * lt**2 * dp_leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :], x=theta)
    return fullMat*2*pi

def gen_mat3_pleg_real(t, lmax, theta, plegs):
    sint = np.sin(theta); cost = np.cos(theta)
    lr = cost; lt = -sint
    #leg, dt_leg, dp_leg = gen_leg_real(lmax, theta)
    leg, dt_leg, dp_leg = plegs
    
    matsize = lmax + 1 - t
    fullMat = np.zeros((matsize*3, matsize*3));
    
    for i in range(matsize*3):
        ui, ess = divmod(i, matsize)
        ess += t
        for j in range(matsize*3):
            uj, ell = divmod(j, matsize)
            ell += t
            if ui==0 and uj==0:
                fullMat[i, j] = simps( sint * lr**2 * leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==0 and uj==1:
                fullMat[i, j] = simps( sint * lr*lt * leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==0 and uj==2:
                fullMat[i, j] = -simps( sint * lr*lt * leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==1 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt * dt_leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==1 and uj==1:
                fullMat[i, j] = simps( sint * lt**2 * dt_leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :], x=theta) 
            elif ui==1 and uj==2:
                fullMat[i, j] = -simps( sint * lt**2 * dt_leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==2 and uj==0:
                fullMat[i, j] = -simps( sint * lr*lt * dp_leg[get_pleg_index(ess, t), :]
                                     * leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==2 and uj==1:
                fullMat[i, j] = -simps( sint * lt**2 * dp_leg[get_pleg_index(ess, t), :]
                                     * dt_leg[get_pleg_index(ell, t), :], x=theta)
            elif ui==2 and uj==2:
                fullMat[i, j] = simps( sint * lt**2 * dp_leg[get_pleg_index(ess, t), :]
                                     * dp_leg[get_pleg_index(ell, t), :], x=theta)
    return fullMat*2*pi
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
    Ashape = A.shape[0]
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

def inv_reg1supp(A, regparam):
    Ashape = A.shape[0]
    iden = np.identity(Ashape)
    regu = iden.copy()
    regv = iden.copy()
    regu[int(Ashape/3):, int(Ashape/3):] = 0
    regv[:int(Ashape/3), :int(Ashape/3)] = 0
    regv[int(Ashape*2/3):, int(Ashape*2/3):] = 0
    return np.linalg.inv(A.transpose().conjugate().dot(A) + regparam * iden
                         - regparam * 0.85 * regv.transpose().dot(regv)
                         + regparam * 7 * regu.transpose().dot(regu)).dot(A.transpose().conjugate())

if __name__=="__main__":
    workingDir = homeDir + "/dopplervel2/matrixA/"
    lmax = 500
    N = int(lmax*1.2)
    theta = np.linspace(1e-5, pi/2 - 1e-5, N)
    pl, dtpleg, dpleg = gen_leg_real(lmax, theta)
    t0 = time.time()

    try:
        procid=int(os.environ['PBS_VNODENUM'])
    except KeyError: pass
    nproc = 48
    tlist = np.arange(procid, lmax+1, nproc)
    print(f" procid = {procid}")
  
    for t in tlist:
        t1 = time.time()
        A = gen_mat3_pleg_real(t, lmax, theta, [pl, dtpleg, dpleg])
        Ainv = inv_reg1supp(A, 1e-3)
        t2 = time.time()
        np.savez_compressed(workingDir + 'A'+str(t).zfill(4)+'.npz', A=A)
        np.savez_compressed(workingDir + 'Ainv'+str(t).zfill(4)+'.npz', A=Ainv)
        print(f"Time taken for t = {t}: {(t2-t1)/60:.3f} min")
    print(f"Total time taken : {(t2-t0)/60:.3f} min")
