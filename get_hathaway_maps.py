"""Construct healPy maps using the synthetic spectrum obtained 
by Hathaway 2015. 
"""
import numpy as np
from astropy.io import fits
import healpy as hp
from globalvars import DopplerVars
import matplotlib.pyplot as plt

gvar = DopplerVars(3)
NSIDE = gvar.nside


def rotate_map(hmap, rot_theta, rot_phi):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)
    return rot_map


def create_los_map(hpmap):
    theta_map, phi_map = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    data_r, data_th, data_ph = hpmap
    losr = np.sin(theta_map) * np.cos(phi_map)
    lost = - np.cos(theta_map) * np.cos(phi_map)
    losp = np.sin(phi_map)
    los_map = losr*data_r + lost*data_th + losp*data_ph
    return los_map

def get_maps(alm):
    ulm, slmp, slmm = alm['ulm'], alm['vlm'], alm['wlm']
    ulm = hp.map2alm(hp.alm2map(ulm, NSIDE))
    slmm = hp.map2alm(hp.alm2map(slmm, NSIDE))
    slmp = hp.map2alm(hp.alm2map(slmp, NSIDE))

    map1r = hp.sphtfunc.alm2map(ulm, NSIDE)
    map1t = hp.sphtfunc.alm2map(slmp, NSIDE)
    map1p = hp.sphtfunc.alm2map(slmm, NSIDE)
#    map1r = rotate_map(map1r, np.pi/2, np.pi)
#    map1t = rotate_map(map1t, np.pi/2, np.pi)
#    map1p = rotate_map(map1p, np.pi/2, np.pi)
    return (map1r, map1t, map1p)

def get_maps_before_inv(ulm, vlm, wlm):
    ulm = hp.map2alm(hp.alm2map(ulm, NSIDE))
    vlm = hp.map2alm(hp.alm2map(vlm, NSIDE))
    wlm = hp.map2alm(hp.alm2map(wlm, NSIDE))

    map1r = hp.sphtfunc.alm2map(ulm, NSIDE)
    map1t = hp.sphtfunc.alm2map(vlm, NSIDE)
    map1p = hp.sphtfunc.alm2map(wlm, NSIDE)

    map1r = rotate_map(map1r, np.pi/2, np.pi)
    map1t = rotate_map(map1t, np.pi/2, np.pi)
    map1p = rotate_map(map1p, np.pi/2, np.pi)
    return (map1r, map1t, map1p)


if __name__ == "__main__":
    alm = np.load("/scratch/g.samarth/HMIDATA/data_analysis/" +
                "lmax1535/alm.data.inv.final018.npz")
    ulm = np.load("/scratch/g.samarth/HMIDATA/data_analysis/ulm003.npy")
    vlm = np.load("/scratch/g.samarth/HMIDATA/data_analysis/vlm003.npy")
    wlm = np.load("/scratch/g.samarth/HMIDATA/data_analysis/wlm003.npy")
    # hp_maps = get_maps(alm)
    hp_maps = get_maps_before_inv(ulm, vlm, wlm)
    los_map = create_los_map(hp_maps)
    hp.mollview(los_map, cmap='seismic')
    plt.show()
