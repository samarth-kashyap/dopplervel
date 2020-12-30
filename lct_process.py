import numpy as np
from astropy.io import fits
import healpy as hp
import argparse
import matplotlib.pyplot as plt
from math import sqrt, pi

NSIDE = 128
print(f"NSIDE = {NSIDE}")


# {{{ reading arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--daynum", help="day number", type=int)
parser.add_argument("--year", help="year", type=int)
parser.add_argument("--testrun", help="run tests", action="store_true")
args = parser.parse_args()
# }}} argparse


# {{{ def make_map(theta, phi, data, NSIDE):
def make_map(theta, phi, data, NSIDE):
    """Makes healpy map given HMI image and coordinate data

    Parameters:
    -----------
    theta - np.ndarray(ndim=1, dtype=np.float)
        latitudes of the data points
    phi - np.ndarray(ndim=1, dtype=np.float)
        longitudes of data points
    data - np.ndarray(ndim=1, dtype=np.float)
        datapoints
    NSIDE - int
        the NSIDE parameter for healPix

    Returns:
    --------
    (map, extmap, thmap, phmap)
    map - healPy map
        healPy map containing the signal
    extmap - healPy mask
        mask containing signal existance pixels
    thmap - healPy map
        coordinate theta - healPy map
    phmap - healPy map
        coorindate phi - healPy map
    """
    assert len(theta) == len(phi) == len(data)
    num_pix = hp.nside2npix(NSIDE)
#    e1map = np.full(num_pix, hp.UNSEEN, dtype=np.float)
    e1map = np.full(num_pix, 0.0, dtype=np.float)
    existance = np.full(num_pix, False, dtype=np.bool)
    counts = np.ones(num_pix, dtype=np.int)
    theta_new = np.zeros(num_pix)
    phi_new = np.zeros(num_pix)

    for i, k in enumerate(data):
        index = hp.ang2pix(NSIDE, theta[i], phi[i])
        theta_new[index], phi_new[index] = hp.pix2ang(NSIDE, index)
        if not existance[index]:
            e1map[index] = 0
            counts[index] = 0
            existance[index] = True
        e1map[index] += k #*np.sin(theta[i])*dtheta_obs*dphi_obs*NSIDE/2.0/np.pi
        counts[index] += 1
    return e1map/counts, existance, theta_new, phi_new
    # return e1map, existance, theta_new, phi_new
# }}} make_map(theta, phi, data, NSIDE)


# {{{ def get_spin1_maps(dat_map, msk_map, th_map, ph_map):
def get_spin1_maps(maps, mask_map):
    """Generates the spin0 and spin1 maps for a given image and coordinates.

    Parameters:
    -----------
    data_map - np.ndarray(ndim=1, dtype=float)
        healPy map of the observed data
    mask_map - np.ndarray(ndim=1, dtype=bool)
        healPy map of the mask

    Returns:
    --------
    map_trans - list
        len(map_trans) = 2
        map_trans[0] - spin1 map corresponding to spin=+1
        map_trans[1] - spin1 map corresponding to spin=-1

    """
    # Finding the vector spherical harmonic coefficients
    map_th, map_ph = maps
    map_trans = [map_th/sqrt(2), map_ph/sqrt(2)]
    map_trans[0][~mask_map] = 0.0
    map_trans[1][~mask_map] = 0.0

    return map_trans
# }}} get_spin1_maps(data_map, mask_map, theta_map, phi_map)


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
        ps[ell] = (abs(coefs[index])**2).sum() * ell * 2
    return np.sqrt(ps)
# }}} computePS(ellArr, emmArr, lmax, coefs)

if __name__ == "__main__":
    data_dir = "/scratch/seismogroup/LCT_data"
    lat_file = f"{data_dir}/{args.year}_lat_files.txt"
    lon_file = f"{data_dir}/{args.year}_lon_files.txt"
    with open(lat_file, "r") as f:
        lat_fnames = f.read().splitlines()

    with open(lon_file, "r") as f:
        lon_fnames = f.read().splitlines()

    latfname = lat_fnames[args.daynum]
    lonfname = lon_fnames[args.daynum]

    print(f"[{args.year}] [{args.daynum}] Loading LCT maps")
    vlat = fits.open(f"{data_dir}/{latfname}")[0].data
    vlon = fits.open(f"{data_dir}/{lonfname}")[0].data

    mask = ~np.isnan(vlat)

    lat_HG = np.linspace(-59.8+90,59.8+90,300)*np.pi/180
    lon_HG = np.linspace(0.2,359.8,900)*np.pi/180

    dtheta_obs, dphi_obs = lat_HG[1] - lat_HG[0], lon_HG[1]-lon_HG[0]
    LON, LAT = np.meshgrid(lon_HG, lat_HG)

    print(f"[{args.year}] [{args.daynum}] Making healPy maps")
    vlat_map, vlat_mask, th_map, ph_map = make_map(LAT[mask],
                                                   LON[mask],
                                                   vlat[mask],
                                                   NSIDE)
    vlon_map, vlon_mask, th_map, ph_map = make_map(LAT[mask],
                                                   LON[mask],
                                                   vlon[mask],
                                                   NSIDE)
    map2trans = get_spin1_maps((vlat_map, vlon_map), vlat_mask)

    print(f"[{args.year}] [{args.daynum}] SHT of healPy maps")
    alm_pm = hp.map2alm_spin(map2trans, 1)
    alm_v = -alm_pm[0]*sqrt(2)
    alm_w = -1j*alm_pm[1]*sqrt(2)
    ellmax = hp.sphtfunc.Alm.getlmax(len(alm_v))
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(ellmax)

    if args.testrun:
        np.savez_compressed(f"/scratch/g.samarth/HMIDATA/" +
                            f"LCT/alm_test.npz",
                            vlm=alm_v, wlm=alm_w)
    else:
        np.savez_compressed(f"/scratch/g.samarth/HMIDATA/" +
                            f"LCT/almo_{args.year}_{args.daynum:03d}.npz",
                            vlm=alm_v, wlm=alm_w)
    #
    if args.daynum==1:
        np.savez_compressed(f"/scratch/g.samarth/HMIDATA/" +
                            f"LCT/arrlm_{args.year}.npz",
                            ellArr=ellArr, emmArr=emmArr)

    print(f"[{args.year}] [{args.daynum}] Plotting and saving")
    psv = computePS(ellArr, emmArr, ellmax, alm_v)
    psw = computePS(ellArr, emmArr, ellmax, alm_w)
    ell = np.arange(ellmax+1)

    plt.figure(figsize=(7, 7))
    plt.loglog(ell, psv, 'r', label='poloidal')
    plt.loglog(ell, psw, 'b', label='toroidal')
    plt.xlabel('Spherical harmonic degree $l$')
    plt.ylabel('Velocity in m/s')
    plt.legend()
    if args.testrun:
        plt.savefig(f"/scratch/g.samarth/HMIDATA/" +
                    f"LCT/lct_sph_test.pdf")
    else:
        plt.savefig(f"/scratch/g.samarth/HMIDATA/" +
                    f"LCT/lct_sph_{args.year}_{args.daynum:03d}.pdf")

