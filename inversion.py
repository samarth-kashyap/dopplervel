from sklearn import linear_model as sklin # for L1 regularization using LASSO
from pyshtools import legendre as pleg    # Legendre polynomials
from scipy.integrate import simps         # Integration - simpsons
import matplotlib.pyplot as plt           # Plotting
from math import sqrt, pi, e              # Math constants
from astropy.io import fits               # FITS handling
import astropy.units as u                 # Handling qty with units
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on daahpc",
                    action="store_true")
parser.add_argument('--cchpc', help="Run program on cchpc19",
                    action="store_true")
parser.add_argument('--gnup', help="Argument for GNU Parallel",
                    type=int)
parser.add_argument('--synth', help="Invert for synthetic data",
                    action="store_true")
parser.add_argument('--read', help="Read matrix from file (for synthetic)",
                    action="store_true")
parser.add_argument('--fat', help="Use fat matrix for inversion",
                    action="store_true")
parser.add_argument('--l1', help="L1 minimization - LASSO",
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

#import sys; sys.path.append(home_dir)
#from heliosPy import datafuncs as cdata
#from heliosPy import mathfuncs as cmath
#from heliosPy import iofuncs as cio

def get_pleg_index(l, m):
    """Gets the index for accessing legendre polynomials 
    (generated from pyshtools.legendre)

    Parameters: 
    -----------
    l - int
        spherical harmonic degree
    m - int
        azimuthal order 

    Returns:
    --------
    index - int
        int( l*(l+1)/2 + m )

    """
    return int(l*(l+1)/2 + m)

def gen_leg(lmax, theta):
    """Generates legendre polynomials and derivatives normalized to 1.0

    Parameters:
    -----------
    lmax - int
        maximum spherical harmonic degree
    theta - np.ndarray(ndim=1, dtype=float)
        array for theta to compute P_l(cos(theta))

    Returns:
    --------
    leg - np.ndarray(ndim=2, dtype=float)
        P_l(cos(theta))
    dt_leg - np.ndarray(ndim=2, dtype=float)
        d/d(theta) P_l(cos(theta)) 
    dp_leg - np.ndarray(ndim=2, dtype=complex)
        im * P_l(cos(theta)) / sin(theta)

    """
    cost = np.cos(theta)
    sint = np.sin(theta)
    maxIndex = int( (lmax+1)*(lmax+2)/2 )
    leg = np.zeros( (maxIndex, theta.size) )
    leg_d1 = np.zeros( (maxIndex, theta.size) )
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
    dp_leg = 1j * emmArr.reshape(maxIndex, 1) \
            * legtemp / sint.reshape(1, sint.shape[0])
    dt_leg /= norm.reshape(maxIndex, 1)
    dp_leg /= norm.reshape(maxIndex, 1)

    return leg/sqrt(2)/sqrt(2*pi), \
            dt_leg/sqrt(2)/sqrt(2*pi), \
            dp_leg/sqrt(2)/sqrt(2*pi)

def gen_leg_real(lmax, theta):
    """Generates legendre polynomials and derivatives normalized to 1.0

    Parameters:
    -----------
    lmax - int
        maximum spherical harmonic degree
    theta - np.ndarray(ndim=1, dtype=float)
        array for theta to compute P_l(cos(theta))

    Returns:
    --------
    leg - np.ndarray(ndim=2, dtype=float)
        P_l(cos(theta))
    dt_leg - np.ndarray(ndim=2, dtype=float)
        d/d(theta) P_l(cos(theta)) 
    dp_leg - np.ndarray(ndim=2, dtype=complex)
        im * P_l(cos(theta)) / sin(theta)

    """
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
    dp_leg = emmArr.reshape(maxIndex, 1) \
            * legtemp / sint.reshape(1, sint.shape[0])
    dt_leg /= norm.reshape(maxIndex, 1)
    dp_leg /= norm.reshape(maxIndex, 1)

    return leg/sqrt(2)/sqrt(2*pi), \
            dt_leg/sqrt(2)/sqrt(2*pi), \
            dp_leg/sqrt(2)/sqrt(2*pi)

def vel_from_spectra_allt(ulm, vlm, wlm, thSize, phSize, lmax):
    """Velocity components in spherical coordinates from the vector spherical
    harmonic coefficients.

    Parameters:
    -----------
    ulm - np.ndarray(ndim=1, dtype=complex)
        radial spherical harmonics
    vlm - np.ndarray(ndim=1, dtype=complex)
        spheroidal spherical harmonics
    wlm - np.ndarray(ndim=1, dtype=complex)
        toroidal spherical harmonics
    thSize - int
        size of theta array
    phSize - int
        size of phi array
    lmax - int

    Returns:
    --------
    ur - np.ndarray(ndim=1, dtype=complex)
        radial component of velocity
    ut - np.ndarray(ndim=1, dtype=complex)
        theta component of velocity
    up - np.ndarray(ndim=1, dtype=complex)
        phi component of velocity

    Notes:
    ------
    The velocity profile is computed using all the spectral coefficients 
    upto lmax.

    """
    theta = np.linspace(1e-5, pi-1e-5, thSize)
    phi = np.linspace(1e-5, 2*pi - 1e-5, phSize)
    
    leg, dt_leg, dp_leg = gen_leg(lmax, theta)
    maxIndex = int( (lmax+1)*(lmax+2)/2 )
    
    ur = np.zeros( (thSize, phSize), dtype=complex)
    ut = np.zeros( (thSize, phSize), dtype=complex)
    up = np.zeros( (thSize, phSize), dtype=complex)
    
    countm, countl = 0, 0
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
    """Generates the full leakage matrix.
    
    Parameters:
    -----------
    t - int
        azimuthal order 
    lmax - int
        maximum spherical harmonic degree
    theta - np.ndarray(ndim=1, type=float)
        latitude array ( pole at disk center )

    Returns:
    --------
    fullMat - np.ndarray(ndim=2, dtype=complex)
        the full leakage matrix

    """
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
                fullMat[i, j] = simps( sint * lr**2 \
                        * leg[get_pleg_index(ess, t), :] \
                        * leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta) 
            elif ui==0 and uj==1:
                fullMat[i, j] = simps( sint * lr*lt \
                        * leg[get_pleg_index(ess, t), :]\
                        * dt_leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta) 
            elif ui==0 and uj==2:
                fullMat[i, j] = simps( sint * lr*lt\
                        * leg[get_pleg_index(ess, t), :]\
                        * dp_leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta)
            elif ui==1 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt\
                        * dt_leg[get_pleg_index(ess, t), :]\
                        * leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta) 
            elif ui==1 and uj==1:
                fullMat[i, j] = simps( sint * lt**2\
                        * dt_leg[get_pleg_index(ess, t), :]
                        * dt_leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta) 
            elif ui==1 and uj==2:
                fullMat[i, j] = simps( sint * lt**2\
                        * dt_leg[get_pleg_index(ess, t), :]
                        * dp_leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta)
            elif ui==2 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt\
                        * dp_leg[get_pleg_index(ess, t), :]
                        * leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta)
            elif ui==2 and uj==1:
                fullMat[i, j] = simps( sint * lt**2\
                        * dp_leg[get_pleg_index(ess, t), :]
                        * dt_leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta)
            elif ui==2 and uj==2:
                fullMat[i, j] = simps( sint * lt**2\
                        * dp_leg[get_pleg_index(ess, t), :]
                        * dp_leg[get_pleg_index(ell, t), :].conjugate(),
                        x=theta)

    return fullMat*2*pi

def gen_full_mat3_real(t, lmax, theta):
    """Generates the full leakage matrix with real components.
    
    Parameters:
    -----------
    t - int
        azimuthal order 
    lmax - int
        maximum spherical harmonic degree
    theta - np.ndarray(ndim=1, type=float)
        latitude array ( pole at disk center )

    Returns:
    --------
    fullMat - np.ndarray(ndim=2, dtype=float)
        the full leakage matrix (with real components)

    """
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
                fullMat[i, j] = simps( sint * lr**2\
                        * leg[get_pleg_index(ess, t), :]\
                        * leg[get_pleg_index(ell, t), :],
                        x=theta) 
            elif ui==0 and uj==1:
                fullMat[i, j] = simps( sint * lr*lt\
                        * leg[get_pleg_index(ess, t), :]\
                        * dt_leg[get_pleg_index(ell, t), :],
                        x=theta) 
            elif ui==0 and uj==2:
                fullMat[i, j] = -simps( sint * lr*lt\
                        * leg[get_pleg_index(ess, t), :]\
                        * dp_leg[get_pleg_index(ell, t), :],
                        x=theta)
            elif ui==1 and uj==0:
                fullMat[i, j] = simps( sint * lr*lt \
                        * dt_leg[get_pleg_index(ess, t), :]\
                        * leg[get_pleg_index(ell, t), :],
                        x=theta) 
            elif ui==1 and uj==1:
                fullMat[i, j] = simps( sint * lt**2 \
                        * dt_leg[get_pleg_index(ess, t), :]\
                        * dt_leg[get_pleg_index(ell, t), :],
                        x=theta) 
            elif ui==1 and uj==2:
                fullMat[i, j] = -simps( sint * lt**2 \
                        * dt_leg[get_pleg_index(ess, t), :]\
                        * dp_leg[get_pleg_index(ell, t), :],
                        x=theta)
            elif ui==2 and uj==0:
                fullMat[i, j] = -simps( sint * lr*lt \
                        * dp_leg[get_pleg_index(ess, t), :]\
                        * leg[get_pleg_index(ell, t), :],
                        x=theta)
            elif ui==2 and uj==1:
                fullMat[i, j] = -simps( sint * lt**2 \
                        * dp_leg[get_pleg_index(ess, t), :]\
                        * dt_leg[get_pleg_index(ell, t), :],
                        x=theta)
            elif ui==2 and uj==2:
                fullMat[i, j] = simps( sint * lt**2 \
                        * dp_leg[get_pleg_index(ess, t), :]\
                        * dp_leg[get_pleg_index(ell, t), :],
                        x=theta)
                
    return fullMat*2*pi

def gen_fat_mat3_real(t, lmax, theta):
    """Generates the fat leakage matrix with real components.
    
    Parameters:
    -----------
    t - int
        azimuthal order 
    lmax - int
        maximum spherical harmonic degree
    theta - np.ndarray(ndim=1, type=float)
        latitude array ( pole at disk center )

    Returns:
    --------
    fatMat - np.ndarray(ndim=2, dtype=complex)
        the fat leakage matrix (with real components)

    Notes:
    ------
    The RHS has spectral coefficients corresponding to (ulm, vlm, wlm) and the
    LHS has spectral coefficients of (ulm) only. Hence the leakage matrix 
    is fat. 

    """
    sint = np.sin(theta)
    cost = np.cos(theta)
    
    lr = cost; lt = -sint
    leg, dt_leg, dp_leg = gen_leg_real(lmax, theta)
    
    matsize = lmax + 1 - t
    fullMat = np.zeros((matsize, matsize*3))
    
    for i in range(matsize):
        ui, ess = divmod(i, matsize)
        ess += t
        for j in range(matsize*3):
            uj, ell = divmod(j, matsize)
            ell += t
            if ui==0 and uj==0:
                fullMat[i, j] = simps( sint * lr*lr \
                        * leg[get_pleg_index(ess, t), :]\
                        * leg[get_pleg_index(ell, t), :],
                        x=theta) 
            elif ui==0 and uj==1:
                fullMat[i, j] = simps( sint * lr*lt \
                        * leg[get_pleg_index(ess, t), :]
                        * dt_leg[get_pleg_index(ell, t), :],
                        x=theta) 
            elif ui==0 and uj==2:
                fullMat[i, j] = -simps( sint * lr*lt \
                        * leg[get_pleg_index(ess, t), :]
                        * dp_leg[get_pleg_index(ell, t), :],
                        x=theta)

    return fullMat*2*pi

def inv_SVD(A, svdlim, plotsigma=False):
    """Computes pseudo-inverse using Singular Value Decomposition after 
    ignoring singular values below given limit.

    Parameters:
    -----------
    A - np.ndarray(ndim=2, dtype=complex)
        the matrix that needs to be inverted

    svdlim - double
        cutoff for inverse of singular values
        e.g. cutoff = 1e4 \implies all elements of \sigma/sinv[0] > 1e4 
        is set to 0

    Returns:
    --------
    Ainv - np.ndarray(ndim=2, dtype=complex)
        pseudo inverse of the matrix A

    
    """
    u, s, v = np.linalg.svd(A, full_matrices=False)
    sinv = s**-1
    if plotsigma:
        plt.figure(); plt.semilogy(sinv, '.')
        plt.axhline(y=svdlim)
        plt.title("Singular values")
        plt.show()
    sinv[sinv/sinv[0] > svdlim] = 0.0#svdlim
    return np.dot( v.transpose().conjugate(),
                   np.dot(np.diag(sinv), u.transpose().conjugate()))

def inv_reg1(A, regparam):
    """Computes the regularized inverse using identity matrix as a 
    regularization.

    Parameters:
    -----------
    A - np.ndarray(ndim=2, dtype=complex)
        the matrix that needs to be inverted
    regparam - float
        parameter for regularization

    Returns:
    --------
    regularized inverse of A
    
    """
    Ashape = A.shape[0]
    return np.linalg.inv(A.transpose().conjugate().dot(A)
        + regparam * np.identity(Ashape)).dot( A.transpose().conjugate())

def inv_reg1supp(A, regparam):
    """Computes the regularized inverse using diagonal matrix as a 
    regularization. Different weights are given to u, v, w.

    Parameters:
    -----------
    A - np.ndarray(ndim=2, dtype=complex)
        the matrix that needs to be inverted
    regparam - float
        parameter for regularization

    Returns:
    --------
    regularized inverse of A
    
    """
    Ashape = A.shape[0]
    iden = np.identity(Ashape)
    regu = iden.copy()
    regv = iden.copy()
    regu[int(Ashape/3):, int(Ashape/3):] = 0
    regv[:int(Ashape/3), :int(Ashape/3)] = 0
    regv[int(Ashape*2/3):, int(Ashape*2/3):] = 0
    return np.linalg.inv(A.transpose().conjugate().dot(A)
                         + regparam * iden
                         - regparam * 0.85 * regv.transpose().dot(regv)
                         + regparam * 7 * regu.transpose().dot(regu))\
                    .dot(A.transpose().conjugate())

def inv_reg2(A, regparam):
    """Computes the regularized inverse using D2 operator as regularization.

    Parameters:
    -----------
    A - np.ndarray(ndim=2, dtype=complex)
        the matrix that needs to be inverted
    regparam - float
        parameter for regularization

    Returns:
    --------
    regularized inverse of A
    
    """
    reg2 = 2*np.identity(A.shape[0])
    offd2 = -1*np.identity(A.shape[0]-1)
    reg2[1:, :-1] += offd2
    reg2[:-1, 1:] += offd2
    reg = reg2[1:-1, :].copy()
    return np.linalg.inv(A.transpose().dot(A)
                         + (regparam/16.) * reg.transpose().dot(reg))\
                    .dot(A.transpose())

def inv_reg3(A, regparam):
    """Computes the regularized inverse using D3 operator as regularization.
    Parameters:
    -----------
    A - np.ndarray(ndim=2, dtype=complex)
        the matrix that needs to be inverted
    regparam - float
        parameter for regularization

    Returns:
    --------
    regularized inverse of A
    
    """
    reg2 = 3*np.identity(A.shape[0])
    offd2 = -1*np.identity(A.shape[0]-1)
    reg2[:-1, 1:] += 3*offd2
    reg2[1:, :-1] += offd2
    reg2[:-2, 2:] += -offd2[1:, 1:]
    reg = reg2[1:-2, :].copy()
    return np.linalg.inv(A.transpose().dot(A)
                         + (regparam/64.) * reg.transpose().dot(reg))\
                    .dot(A.transpose())

def get_only_t(lmax, t, alm, ellArr, emmArr):
    """Filter out the spectral coefficients corresponding to a given 
    azimuthal order t.
    
    Parameters:
    ----------
    lmax - int
        max spherical harmonic degree
    t - int
        azimuthal order to be filtered
    alm - np.ndarray(ndim=1, dtype=complex)
        the spectral coefficients
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing list of ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing list of emm values

    Returns:
    --------
    alm - np.ndarray(ndim=1, dtype=complex)
        the spectral coefficients corresponding to the given t

    """
    ist = emmArr==t
    isel = ( (ellArr>=t) * (ellArr<=lmax) )
    mask = ist * isel
    return alm[mask]

def put_only_t(lmax, t, alm, almFull, ellArr, emmArr):
    """Updates the full array of spectral coefficients with the 
    coefficients of a given t.
    
    Parameters:
    ----------
    lmax - int
        max spherical harmonic degree
    t - int
        azimuthal order to be filtered
    alm - np.ndarray(ndim=1, dtype=complex)
        the array of spectral coefficients (for a given t)
    almFull - np.ndarray(ndim=1, dtype=complex)
        the full array of all spectral coefficients (upto lmax)
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing list of ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing list of emm values

    Returns:
    --------
    almFull - np.ndarray(ndim=1, dtype=complex)
        the full array of all spectral coefficients (upto lmax)

    """
    ist = emmArr==t
    isel = ( (ellArr>=t) * (ellArr<=lmax) )
    mask = ist * isel
    almFull[mask] = alm
    return almFull

def deconcat(alm):
    '''Deconcatenates full array into the componets of ulm, vlm, wlm.
    
    Parameters:
    -----------
    alm - np.ndarray(ndim=1, dtype=complex)
        the concatenated array

    Returns:
    --------
    alm1 - np.ndarray(ndim=1, dtype=complex)
       ulm array
    alm2 - np.ndarray(ndim=1, dtype=complex)
       vlm array
    alm3 - np.ndarray(ndim=1, dtype=complex)
       wlm array

    '''
    totsize = alm.shape[0]
    deconsize = int(totsize/3)
    alm1 = alm[:deconsize]
    alm2 = alm[deconsize:2*deconsize]
    alm3 = alm[2*deconsize:]
    return alm1, alm2, alm3

def computePS(alm, lmax, ellArr, emmArr):
    '''Computes the power spectrum given the spectral coefficients.

    Parameters:
    -----------
    alm - np.ndarray(ndim=1, dtype=complex)
        array of all spectral coefficients (upto lmax)
    lmax - int 
        maximum spherical harmonic degree
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing ell 
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing emm
    
    Returns:
    --------
    ps - np.ndarray(ndim=1, dtype=float)
        power spectrum 
    
    Notes:
    ------
    The velocity power spectrum is given by ell * \sum_{m} | alm |^2
    
    '''
    ps = np.zeros(lmax)
    for i in range(lmax):
        isel = ellArr==i
        ps[i] += (abs(alm[isel])**2).sum() * i# / (2*i + 1)
    return np.sqrt(ps)

if __name__=="__main__":
    print("loading files --")
    workingDir = scratch_dir + "matrixA/"
    data_dir = scratch_dir + "HMIDATA/data_analysis/"

    if args.synth:
        # synthetic data has both true and observed alms
        arrFile = np.load("arrlm.npz")# contains arrays of ell and emm
        almFile = np.load("alm.npz")  # contains arrays of alms - observed
        almAFile = np.load("almA.npz")# contains arrays of alms - theoretical
        # loading ell and emm arrays
        ellArr = arrFile['ellArr']
        emmArr = arrFile['emmArr']
        # loading alms - observed
        ulmo = almFile['ulm']
        vlmo = almFile['vlm']
        wlmo = almFile['wlm']
        # loading alms - theoretical
        ulmAth = almAFile['ulm']
        vlmAth = almAFile['vlm']
        wlmAth = almAFile['wlm']
    else:
        # real data has only observed alms
        ellArr = np.load(data_dir + "ellArr.txt.npz")['ellArr']
        emmArr = np.load(data_dir + "emmArr.txt.npz")['emmArr']
        # loading alms - observed
        if args.gnup:
            suffix = str(args.gnup).zfill(3) + ".txt.npz"
        else:
            suffix = ".txt.npz"
        ulmo = np.load(data_dir + "ulm" + suffix)['ulm']
        vlmo = np.load(data_dir + "vlm" + suffix)['vlm']
        wlmo = np.load(data_dir + "wlm" + suffix)['wlm']

    print("loading files -- complete")

    ulmA = np.zeros(ulmo.shape[0], dtype=complex)
    vlmA = np.zeros(ulmo.shape[0], dtype=complex)
    wlmA = np.zeros(ulmo.shape[0], dtype=complex)

    lmaxData = int(ellArr.max())
    lmaxCalc = 500

    ell = np.arange(lmaxCalc)
    thSize = int(lmaxCalc * 1.2)
    phSize = 2*thSize
    theta = np.linspace(1e-5, pi/2 - 1e-5, thSize)
    phi = np.linspace(1e-5, 2*pi - 1e-5, phSize)

    r = np.sin(theta).reshape(thSize, 1)
    x = r * np.cos(phi).reshape(1, phSize)
    y = r * np.sin(phi).reshape(1, phSize)

    print(f"lmax = {lmaxCalc}, thSize = {thSize}")

    print("computing for .... ")
    t0 = time.time()

    for t in range(lmaxCalc):
        t1 = time.time()
        if args.synth:
            if args.fat:
                if args.read:
                    A = np.load(workingDir
                                + "fatMat"+str(t).zfill(2)+".npz")['A']
                else:
                    A = gen_fat_mat3_real(t, lmaxCalc, theta)
                    np.savez_compressed(workingDir
                                        + "fatMat"+str(t).zfill(2)+".npz",
                                        A=A)
                Ainv = inv_SVD(A, 1e4)
            else:
                if args.read:
                    A = np.load(workingDir
                                + "fullMat"+str(t).zfill(2)+".npz")['A']
                else:
                    A = gen_full_mat3_real(t, lmaxCalc, theta)
                    np.savez_compressed(workingDir
                                        + "fullMat"+str(t).zfill(2)+".npz",
                                        A=A)
                Ainv = inv_reg1supp(A, 1e-3)
        else:
            """
            if args.fat:
                if args.read:
                    A = np.load(workingDir
                                + "fatMat"+str(t).zfill(2)+".npz")['A']
                else:
                    A = gen_fat_mat3_real(t, lmaxCalc, theta)
                    np.savez_compressed(workingDir
                                        + "fatMat"+str(t).zfill(2)+".npz",
                                        A=A)
                Ainv = inv_SVD(A, 1e4)
            else:
                if args.read:
                    A = np.load(workingDir
                                + "fullMat"+str(t).zfill(2)+".npz")['A']
                else:
                    A = gen_full_mat3_real(t, lmaxCalc, theta)
                    np.savez_compressed(workingDir
                                        + "fullMat"+str(t).zfill(2)+".npz",
                                        A=A)
                Ainv = inv_reg1supp(A, 1e-3)
            """
            A = np.load(workingDir + "A"+str(t).zfill(4)+".npz")['A']
            Ainv = np.load(workingDir + "Ainv"+str(t).zfill(4)+".npz")['A']
#        Ainv = inv_SVD(A, 1e4)

        ulmt = get_only_t(lmaxCalc, t, ulmo, ellArr, emmArr)
        vlmt = get_only_t(lmaxCalc, t, vlmo, ellArr, emmArr)
        wlmt = get_only_t(lmaxCalc, t, wlmo, ellArr, emmArr)

        if args.fat:
            uot = ulmt
        else:
            uot = np.concatenate((ulmt, vlmt, 1j*wlmt), axis=0)

        assert uot.shape[0] == A.shape[0] == Ainv.shape[0]

        if args.l1:
            clf = sklin.Lasso(alpha=1e-4, max_iter=10000)
            clf.fit(A, uot.real)
            uAt1 = clf.coef_
            clf.fit(A, uot.imag)
            uAt2 = clf.coef_
            uAt = uAt1 + 1j*uAt2
            del uAt1, uAt2
            t2 = clf.intercept_
        else:
            uAt = Ainv.dot(uot)
        """
        A2 = A.dot(Ainv)
        m1 = -A2 + np.identity(A2.shape[0])
        x0 = Ainv.dot( m1.dot(uot) )
        x1 = m1.dot( x0 )

        uAt += x0 + x1
        """
        uA = deconcat(uAt)

        ulmA = put_only_t(lmaxCalc, t, uA[0], ulmA, ellArr, emmArr)
        vlmA = put_only_t(lmaxCalc, t, uA[1], vlmA, ellArr, emmArr)
        wlmA = put_only_t(lmaxCalc, t, uA[2], 1j*wlmA, ellArr, emmArr)
        
        t2 = time.time()
        if t%20==0:
            print(f"Time taken for t = {t}: {(t2-t1)/60:.3f} min,"\
                  + f"({(t2-t0)/60:.3f} min)")
    tn = time.time()
    print(f"Total time taken = {(tn-t0)/60:.3f} minutes")

    if args.synth:
        np.savez("almInv.npz", ulm=ulmA, vlm=vlmA, wlm=wlmA)
    else:
        if args.gnup:
            suffix = str(args.gnup).zfill(3) + ".npz"
        else:
            suffix = ".npz"
        fname = data_dir + "alm.data.inv" + suffix
        np.savez(fname, ulm=ulmA, vlm=vlmA, wlm=wlmA)

    psu = computePS(ulmA, lmaxCalc, ellArr, emmArr)
    psv = computePS(vlmA, lmaxCalc, ellArr, emmArr)
    psw = computePS(wlmA, lmaxCalc, ellArr, emmArr)
    np.savez_compressed(data_dir + "power.rot" + suffix,
                        upow=psu, vpow=psv, wpow=psw)
    pstot = np.sqrt( psu**2 + psv**2 + psw**2 )

#    urA, utA, upA = vel_from_spectra_allt(ulmA, vlmA, wlmA,
#                                    thSize, phSize, lmaxCalc)

    if not args.synth:
        plt.figure()
        plt.loglog(psu, 'g', label='radial')
        plt.loglog(psv, 'r', label='poloidal')
        plt.loglog(psw, 'b', label='toroidal')
        plt.xlabel('$l$')
        plt.ylabel('velocity $ms^{-1}$')
        plt.legend()
        plt.show()
    
    if args.synth:
        psuth = computePS(ulmAth, lmaxCalc, ellArr, emmArr)
        psvth = computePS(vlmAth, lmaxCalc, ellArr, emmArr)
        pswth = computePS(wlmAth, lmaxCalc, ellArr, emmArr)
        pstotth = np.sqrt( psuth**2 + psvth**2 + pswth**2 )

        plt.figure()
        plt.subplot(221)
        plt.loglog(np.sqrt( ell * psu), 'g-.', label='ulm - inv')
        plt.loglog(np.sqrt( ell * psuth), 'g', label='ulm - actual')
        plt.legend()

        plt.subplot(222)
        plt.loglog(np.sqrt( ell * psv), 'r-.', label='vlm - inv')
        plt.loglog(np.sqrt( ell * psvth), 'r', label='vlm - actual')
        plt.legend()
        
        plt.subplot(223)
        plt.loglog(np.sqrt( ell * psw), 'b-.', label='wlm - inv')
        plt.loglog(np.sqrt( ell * pswth), 'b', label='wlm - actual')
        plt.legend()
        
        plt.subplot(224)
        plt.loglog(np.sqrt( ell * pstot), color='black',
                   linestyle='-.', label='full - inv')
        plt.loglog(np.sqrt( ell * pstotth), 'black', label='full - actual')
        plt.legend()
        plt.show()
        """
        phSize = 2*thSize
        urA, utA, upA = vel_from_spectra_allt(ulmA, vlmA, wlmA, 
                                        thSize, phSize, lmaxCalc)
        urAth, utAth, upAth = vel_from_spectra_allt(ulmAth, vlmAth, 
                                        wlmAth, thSize, phSize, lmaxCalc)

        lr = np.cos(theta).reshape(thSize, 1)
        lt = -np.sin(theta).reshape(thSize, 1)
        losVelA = urA * lr + utA * lt
        losVelAth = urAth * lr + utAth * lt

        np.savetxt("psu.txt", psu)
        np.savetxt("psv.txt", psv)
        np.savetxt("psw.txt", psw)
        np.savetxt("psuth.txt", psuth)
        np.savetxt("psvth.txt", psvth)
        np.savetxt("pswth.txt", pswth)

        np.savez("urtpA.npz", ur=urA, ut=utA, up=upA)
        np.savez("urtpAth.npz", ur=urAth, ut=utAth, up=upAth)

        _max1 = max( abs(urA.real).max(), abs(urA.real).min(), \
            abs(urAth.real).max(), abs(urAth.real).min())
        _max2 = max( abs(utA.real).max(), abs(utA.real).min(), \
            abs(utAth.real).max(), abs(utAth.real).min())
        print(f"x = {x.shape}, y = {y.shape}, ur = {urA.shape}")
        Ncont = 15
        plt.figure()
        plt.subplot(321)
    #    im = plt.imshow(urA.real, cmap='seismic', extent=[0,2*np.pi,pi/2,0])#, vmax=_max1, vmin=-_max1)
        im = plt.contour(x, y, urA.real, Ncont, cmap='seismic', linewidth=0.4)#, vmax=_max1, vmin=-_max1)
        plt.colorbar(im)
        plt.title('$u_r$- inv')
        plt.axis('equal')
        #plt.ylabel(r'$\theta$')
        #plt.xlabel('$\phi$')
        plt.subplot(322)
        #im = plt.imshow(urAth.real, cmap='seismic', extent=[0,2*np.pi,pi/2,0])#, vmax=_max1, vmin=-_max1)
        im = plt.contour(x, y, urAth.real, Ncont, cmap='seismic', linewidth=0.4)#, vmax=_max1, vmin=-_max1)
        plt.colorbar(im)
        plt.title('$u_r$- actual')
        plt.axis('equal')
        #plt.ylabel(r'$\theta$')
        #plt.xlabel('$\phi$')
        plt.subplot(323)
        #im = plt.imshow(utA.real, cmap='seismic', extent=[0,2*np.pi,pi/2,0])#, vmax=_max2, vmin=-_max2)
        im = plt.contour(x, y, utA.real, Ncont, cmap='seismic', linewidth=0.4)#, vmax=_max1, vmin=-_max1)
        plt.colorbar(im)
        plt.title(r'$u_\theta$- inv')
        plt.axis('equal')
        #plt.ylabel(r'$\theta$')
        #plt.xlabel('$\phi$')
        plt.subplot(324)
        #im = plt.imshow(utAth.real, cmap='seismic', extent=[0,2*np.pi,pi/2,0])#, vmax=_max2, vmin=-_max2)
        im = plt.contour(x, y, utAth.real, Ncont, cmap='seismic', linewidth=0.4)#, vmax=_max1, vmin=-_max1)
        plt.colorbar(im)
        plt.title(r'$u_\theta$- actual')
        plt.axis('equal')
        #plt.ylabel(r'$\theta$')
        #plt.xlabel('$\phi$')
        plt.subplot(325)
        #im = plt.imshow(upA.real, cmap='seismic', extent=[0,2*np.pi,pi/2,0])#, vmax=_max2, vmin=-_max2)
        im = plt.contour(x, y, upA.real, Ncont, cmap='seismic', linewidth=0.4)#, vmax=_max1, vmin=-_max1)
        plt.colorbar(im)
        plt.title('$u_\phi$ - inv')
        plt.axis('equal')
        #plt.ylabel(r'$\theta$')
        #plt.xlabel('$\phi$')
        plt.subplot(326)
        #im = plt.imshow(upAth.real, cmap='seismic', extent=[0,2*np.pi,pi/2,0])#, vmax=_max2, vmin=-_max2)
        im = plt.contour(x, y, upAth.real, Ncont, cmap='seismic', linewidth=0.4)#, vmax=_max1, vmin=-_max1)
        plt.colorbar(im)
        plt.title('$u_\phi$ - actual')
        plt.axis('equal')
        #plt.ylabel(r'$\theta$')
        #plt.xlabel('$\phi$')
        plt.tight_layout()
        plt.show()


        plt.figure()

        plt.subplot(231)
        im = plt.contour(x, y, losVelA.real, Ncont, cmap='seismic')
        plt.colorbar(im)
        plt.axis('equal')
        plt.title('los(real) - inverted')
        
        plt.subplot(232)
        im = plt.contour(x, y, losVelAth.real, Ncont, cmap='seismic')
        plt.colorbar(im)
        plt.axis('equal')
        plt.title('los(real) - actual')

        plt.subplot(233)
        im = plt.contour(x, y, losVelA.real + losVelAth.real, Ncont, cmap='seismic')
        plt.colorbar(im)
        plt.axis('equal')
        plt.title('los(real) - diff')

        plt.subplot(234)
        im = plt.contour(x, y, losVelA.imag, Ncont, cmap='seismic')
        plt.colorbar(im)
        plt.axis('equal')
        plt.title('los(imag) - inverted')
        
        plt.subplot(235)
        im = plt.contour(x, y, losVelAth.imag, Ncont, cmap='seismic')
        plt.colorbar(im)
        plt.axis('equal')
        plt.title('los(imag) - actual')

        plt.subplot(236)
        im = plt.contour(x, y, losVelA.imag + losVelAth.imag, Ncont, cmap='seismic')
        plt.colorbar(im)
        plt.axis('equal')
        plt.title('los(imag) - diff')

        plt.tight_layout()
        plt.show()
        """
