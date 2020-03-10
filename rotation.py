from scipy.spatial.transform import Rotation as R
from pyshtools import legendre as pleg     # Legendre polynomials
from scipy.integrate import simps          # Integration - simpsons
import matplotlib.pyplot as plt            # Plotting
from math import sqrt, pi, e               # Math constants
from astropy.io import fits                # FITS handling
import astropy.units as u                  # Handling qty with units
import healpy as hp
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

def alm2almhp(alm, ellArr, emmArr, lmax):
    '''Converts the spectral coefficients array from the pyshtools format
    to the healPy format.
    
    Parameters:
    -----------
    alm - np.ndarray(ndim=1, dtype=complex)
        array of spectral coefficients
    ellArr - np.ndarray(ndim=1, dtype=int)
        array of spherical harmonic degree
    emmArr - np.ndarray(ndim=1, dtype=int)
        arry of azimuthal order
    lmax - int
        maximum spherical harmonic degree

    Returns:
    --------
    almhp - np.ndarray(ndim=1, dtype=complex)
        spectral coefficients in the healPy format

    '''
    _maxind = int((lmax+1)*(lmax+2)/2)
    alm = alm[:_maxind].copy()
    almhp = np.array([], dtype=complex)
    within_max = ellArr<=lmax
    for emm in range(lmax+1):
        isem = emmArr==emm
        mask = within_max * isem
        almhp = np.append(almhp, alm[mask])
    return almhp

def rotate_map_spin_eul(hmap, eulAngle):
    """Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    npix = len(hmap[0])
    nside = hp.npix2nside(npix)

    # Get theta, phi for non-rotated map
    theta, phi = hp.pix2ang(nside, np.arange(npix)) 
    costh, cosph = np.cos(theta), np.cos(phi)
    sinth, sinph = np.sin(theta), np.sin(phi)
    vth = np.array([costh*cosph, costh*sinph, -sinth])
    vph = np.array([-sinph, cosph, 0.0*cosph])
    
    r = hp.rotator.Rotator(eulAngle, deg=False, eulertype='zxz')

    # Get theta, phi under rotated co-ordinates
    theta_rot, phi_rot = r(theta, phi)
    costh_rot, cosph_rot = np.cos(theta_rot), np.cos(phi_rot)
    sinth_rot, sinph_rot = np.sin(theta_rot), np.sin(phi_rot)
    vth_rot = np.array([costh_rot*cosph_rot, costh_rot*sinph_rot, -sinth_rot])
    vph_rot = np.array([-sinph_rot, cosph_rot, 0.0*cosph_rot])

    rotMat = R.from_euler('zxz', eulAngle).as_dcm()
    vth_rot = vth_rot.transpose().dot(rotMat).transpose()
    vph_rot = vph_rot.transpose().dot(rotMat).transpose()

    # Interpolate map onto these co-ordinates
    rot_map0temp = hp.get_interp_val(hmap[0], theta_rot, phi_rot)
    rot_map1temp = hp.get_interp_val(hmap[1], theta_rot, phi_rot)

    # Obtaining the rotated maps
    rot_map0 = (vth*vph_rot).sum(axis=0) * rot_map0temp \
             + (vph*vph_rot).sum(axis=0) * rot_map1temp

    rot_map1 = (vth*vth_rot).sum(axis=0) * rot_map0temp \
             + (vph*vth_rot).sum(axis=0) * rot_map1temp

    return rot_map0, rot_map1

def get_spin1_alms(map_r, map_trans):
    """Get the vector spherical harmonic coefficients for spin1 harmonics.

    Parameters:
    -----------
    map_r - np.ndarray(ndim=1, dtype=float)
        map containing radial component of vector field
    map_trans - list
        len(map_trans) = 2
        map_trans[0] - map of vector field corresponding to +1 component
        map_trans[1] - map of vector field corresponding to -1 component
    
    Returns:
    --------
    alm2r - spin0 spherical harmonic coefficients
    alm2v - spin1 spherical harmonic coefficients for s=+1
    alm2w - spin1 spherical harmonic coefficients for s=-1
    
    """
    assert len(map_r) == len(map_trans[0]) == len(map_trans[1])
    alm_r = hp.map2alm(map_r)
    alm_pm = hp.map2alm_spin(map_trans, 1)
    alm_v = -alm_pm[0]
    alm_w = -1j*alm_pm[1]
    return alm_r, alm_v, alm_w

def computePS(ellArr, emmArr, lmax, coefs):
    """Computes the power spectrum
    
    Parameters:
    -----------
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing the ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing the emm values
    lmax - int
        maximum value of ell
    coefs - np.ndarray(ndim=1, dtype=complex)
        the alm coefficients

    Returns:
    --------
    ps - np.ndarray(ndim=1, dtype=float)
        power spectrum of the given alm

    Notes:
    ------
    Power spectrum = \sum_{m} | alm |^2 

    """
    ps = np.zeros(lmax+1)
    for ell in range(lmax+1):
        isel = ellArr==ell
        ps[ell] = ell * (abs(coefs[isel])**2).sum()# / (2*ell+1)
    return np.sqrt(ps)

def get_ell_emm_arr(lmax):
    '''Returns the ellArr and emmArr arrays for the pyshtools convention.
    
    Parameters:
    -----------
    lmax - int 
        maximum spherical harmonic degree

    Returns:
    --------
    ellArr - np.ndarray(ndim=1, dtype=int)
        array of ell (corresponding to the spectral coefficients)
    emmArr - np.ndarray(ndim=1, dtype=int)
        array of emm (corresponding to the spectral coefficients)

    '''
    ellArr = np.array([], dtype=int)
    emmArr = np.array([], dtype=int)
    for ell in range(lmax+1):
        for emm in range(ell+1):
            ellArr = np.append(ellArr, ell)
            emmArr = np.append(emmArr, emm)
    return ellArr, emmArr

def alm4nside(ulm, vlm, wlm, ellArr, emmArr, nside):
    '''Get alm for given nside
    
    Parameters:
    -----------
    ulm - np.ndarray(ndim=1, dtype=complex)
        radial harmonic coefficients
    vlm - np.ndarray(ndim=1, dtype=complex)
        poloidal harmonic coefficients
    wlm - np.ndarray(ndim=1, dtype=complex)
        toroidal harmonic coefficients
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing ell 
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing emm
    nside - int 
        NSIDE parameter of the healPy map 

    Returns:
    --------
    ulm - np.ndarray(ndim=1, dtype=complex)
        radial harmonic coefficients
    vlm - np.ndarray(ndim=1, dtype=complex)
        poloidal harmonic coefficients
    wlm - np.ndarray(ndim=1, dtype=complex)
        toroidal harmonic coefficients
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing ell 
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing emm
    
    '''
    lmax4nside = 3*nside - 1
    maskell = ellArr<=lmax4nside
    ellArr = ellArr[maskell].copy()
    emmArr = emmArr[maskell].copy()
    ulm = ulm[maskell].copy()
    vlm = vlm[maskell].copy()
    wlm = wlm[maskell].copy()

    return ulm, vlm, wlm, ellArr, emmArr

if __name__=="__main__":
    workingDir = scratch_dir + "matrixA/"
    data_dir = scratch_dir + "HMIDATA/data_analysis/"
    th_dir = home_dir + "dopplervel2/"
    if args.gnup:
        suffix = str(args.gnup).zfill(3) + ".npz"
    else:
        suffix = ".npz"
    fname = data_dir + "alm.data.inv" + suffix
    alm = np.load(fname)
    ellArr = np.load(data_dir + "ellArr.txt.npz")['ellArr']
    emmArr = np.load(data_dir + "ellArr.txt.npz")['ellArr']
    ulm2 = alm['ulm']
    vlm2 = alm['vlm']
    wlm2 = alm['wlm']

    lmax_calc = 500
    NSIDE = 128
    LMAXHP = 3*NSIDE - 1
    SPIN = 1
#    ellArr, emmArr = get_ell_emm_arr(LMAXHP)
#    ulm2 = alm2almhp(ulm2, ellArr, emmArr, LMAXHP)
#    vlm2 = alm2almhp(vlm2, ellArr, emmArr, LMAXHP)
#    wlm2 = alm2almhp(wlm2, ellArr, emmArr, LMAXHP)

    ulm2, vlm2, wlm2, ellArr, emmArr = alm4nside(ulm2, vlm2, wlm2, ellArr,
                                                 emmArr, NSIDE)


    map2r = hp.alm2map(ulm2, NSIDE)
    map2trans = hp.alm2map_spin((-vlm2, 1j*wlm2), NSIDE, SPIN, LMAXHP)

    """
    hp.mollview(map2r, cmap='seismic')
    hp.mollview(map2trans[0], cmap='seismic')
    hp.mollview(map2trans[1], cmap='seismic')
    plt.show()
    """
    eul_angle = np.array([0, -pi/2, 0])

    map1trans = rotate_map_spin_eul(map2trans, eul_angle)
    map1r = map2r.copy()

    ulm1, vlm1, wlm1 = get_spin1_alms(map1r, map1trans)
    ellmax = hp.sphtfunc.Alm.getlmax(len(vlm1))
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(ellmax)
    fname = data_dir + "alm.data.inv.final" + suffix
    np.savez_compressed(fname, ulm=ulm1, vlm=vlm1, wlm=wlm1,
                        NSIDE=NSIDE, ellmax=ellmax)

    upow = computePS(ellArr, emmArr, ellmax, ulm1)
    vpow = computePS(ellArr, emmArr, ellmax, vlm1)
    wpow = computePS(ellArr, emmArr, ellmax, wlm1)
    np.savez_compressed(data_dir+"power.final"+suffix, upow=upow,
                        vpow=vpow, wpow=wpow)
    _max_plot = len(upow)

    uth = np.loadtxt(th_dir + "green.csv", delimiter=",")
    vth = np.loadtxt(th_dir + "red.csv", delimiter=",")
    wth = np.loadtxt(th_dir + "blue.csv", delimiter=",")

    fac = 2.8
    plt.figure()
    plt.loglog(uth[:, 0], uth[:, 1],'g', label='radial')
    plt.loglog(vth[:, 0], vth[:, 1],'r', label='poloidal')
    plt.loglog(wth[:, 0], wth[:, 1],'b', label='toroidal')
    plt.loglog(fac*upow, '--g')
    plt.loglog(fac*vpow, '--r')
    plt.loglog(fac*wpow, '--b')
    plt.legend()
    plt.savefig(data_dir + "ps.png")
    plt.show()
