"""Module for handling spherical harmonic conventions and transformations.

This module provides utilities for working with HEALPix maps and vector spherical
harmonics. It includes functions for generating synthetic maps, computing spin-1
transformations, and calculating power spectra.
"""

import numpy as np
import healpy as hp

# {{{ Global variables
NSIDE = 128
lmax = 3*NSIDE - 1
ellArr, emmArr = hp.sphtfunc.Alm.getlm(lmax)
NPIX = hp.nside2npix(NSIDE)
# }}} global vars


# {{{ def get_synth_maps():
def get_synth_maps():
    """Generate synthetic HEALPix maps for testing.
    
    Creates synthetic velocity field maps with random spherical harmonic
    coefficients for radial and horizontal components. The maps are made
    real by removing imaginary parts through a forward-backward transformation.
    
    Returns
    -------
    r_map : np.ndarray
        HEALPix map of the radial velocity component
    h1map : np.ndarray
        HEALPix map of the first horizontal component
    h2map : np.ndarray
        HEALPix map of the second horizontal component
    """
    ulm_t1 = np.sqrt((np.random.rand(len(ellArr)) - 0.5 +
                      1j*(np.random.rand(len(ellArr)) - 0.5)) *
                     (lmax**2 - (ellArr-50)**2))
    vlm_t1 = np.sqrt((np.random.rand(len(ellArr)) - 0.5 +
                      1j*(np.random.rand(len(ellArr)) - 0.5)) *
                     (lmax**2*1.5 - (ellArr-50)**2))
    wlm_t1 = np.sqrt((np.random.rand(len(ellArr)) - 0.5 +
                      1j*(np.random.rand(len(ellArr)) - 0.5)) *
                     (lmax**2*1.7 - (ellArr-50)**2))

    r_t1 = hp.alm2map(ulm_t1, NSIDE)
    ulm_t1_imag = hp.map2alm(r_t1.imag)
    ulm_t2 = ulm_t1 - 1j*ulm_t1_imag

    r1map = hp.alm2map(ulm_t2, NSIDE)
    r_map = hp.alm2map(hp.map2alm(r1map), NSIDE)

    hlmp_t1 = (vlm_t1 + 1j*wlm_t1)/np.sqrt(2)
    hlmm_t1 = (vlm_t1 - 1j*wlm_t1)/np.sqrt(2)
    t1map, p1map = hp.alm2map_spin((hlmp_t1, hlmm_t1), NSIDE, 1, lmax)

    hlmp_t1_imag, hlmm_t1_imag = hp.map2alm_spin((t1map.imag, p1map.imag), 1)
    hlmp = hlmp_t1 - 1j*hlmp_t1_imag
    hlmm = hlmm_t1 - 1j*hlmm_t1_imag

    t1map, p1map = hp.alm2map_spin((hlmm, hlmp), NSIDE, 1, lmax)
    h1map, h2map = hp.alm2map_spin(hp.map2alm_spin((t1map, p1map), 1), NSIDE, 1, lmax)
    return r_map, h1map, h2map
# }}} get_synth_maps()


# {{{ def get_spin1_maps(dat_map, msk_map, th_map, ph_map, pole="diskCenter"):
def get_spin1_maps(ipmaps):
    """Generates the spin0 and spin1 maps for a given image and coordinates.

    Parameters:
    -----------
    data_map - np.ndarray(ndim=1, dtype=float)
        healPy map of the observed data
    mask_map - np.ndarray(ndim=1, dtype=bool)
        healPy map of the mask
    theta_map - np.ndarray(ndim=1, dtype=float)
        healPy map of the latitudes
    phi_map - np.ndarray(ndim=1, dtype=float)
        healPy map of the longitudes
    pole - string (default = "diskCenter")
        coordinate system pole selector
        allowed - "diskCenter" and "solarNorth"

    Returns:
    --------
    map_r - np.ndarray(ndim=1, dtype=float)
        spin0 map
    map_trans - list
        len(map_trans) = 2
        map_trans[0] - spin1 map corresponding to spin=+1
        map_trans[1] - spin1 map corresponding to spin=-1

    """
    # Finding the vector spherical harmonic coefficients
    rmap, tmap, pmap = ipmaps
    map_r = rmap
    map_p = - (tmap + 1j*pmap) / sqrt(2)
    map_m = - (tmap - 1j*pmap) / sqrt(2)
    map_trans = [(map_p + map_m)/2, -1j*(map_p - map_m)/2]

    map_r[~mask_map] = hp.UNSEEN
    map_trans[0][~mask_map] = 0.0
    map_trans[1][~mask_map] = 0.0

    return map_r, map_trans
# }}} get_spin1_maps(data_map, mask_map, theta_map, phi_map, pole="diskCenter")


# {{{ def get_spin1_alms(map_r, map_trans):
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
    # alm_v = -alm_pm[0]
    # alm_w = -1j*alm_pm[1]
    # return alm_r, alm_v, alm_w
    return alm_r, alm_pm[0], alm_pm[1]
# }}} get_spin1_alms(map_r, map_trans)


# {{{ def computePS(ellArr, emmArr, lmax, coefs):
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
    Power spectrum = sum_{m} | alm |^2

    """
    ps = np.zeros(lmax+1)
    for ell in range(lmax+1):
        index = np.where((ellArr == ell))[0]
        ps[ell] = (abs(coefs[index])**2).sum()*2  # / (2*ell+1)
    return ps
# }}} computePS(ellArr, emmArr, lmax, coefs)


if __name__ == "__main__":
    rmap, tmap, pmap = get_synth_maps()
    map_r, map_trans = get_spin1_maps((rmap, tmap, pmap))
    ulm, ulmp, ulmm = get_spin1_alms(map_r, map_trans)
    vlm, wlm = np.sqrt(2)*ulmp, np.sqrt(2)*ulmm

    pu0 = computePS(ellArr, emmArr, lmax, ulm)
    pv0 = computePS(ellArr, emmArr, lmax, vlm)
    pw0 = computePS(ellArr, emmArr, lmax, wlm)



    
