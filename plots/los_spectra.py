"""Construct the power spectrum using LOS Dopplergrams. 
- Creates healPy maps from the observed HMI data
- computes power spectrum from alm of healPy maps
"""
import numpy as np
from astropy.io import fits
import healpy as hp
import matplotlib.pyplot as plt
from globalvars import DopplerVars
import argparse
import time


# {{{ Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--gnup', help="Argument for gnuParallel",
                    type=int)
args = parser.parse_args()
gvar = DopplerVars(args.gnup)
NSIDE = gvar.nside
# }}} parser


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
    e1map = np.full(num_pix, hp.UNSEEN, dtype=np.float)
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
        e1map[index] += k
        counts[index] += 1
    return e1map/counts, existance, theta_new, phi_new
# }}} make_map(theta, phi, data, NSIDE)


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
        maskell = ellArr == ell
        ps[ell] = ell * ((abs(coefs[maskell])**2).sum())
    return np.sqrt(ps)
# }}} computePS(ellArr, emmArr, lmax, coefs)




if __name__ == "__main__":
    rw_dir = gvar.outdir
    # suffix convention
    # 1 - used for coordinates where pole is at solar north pole
    # 2 - used for coordinates where pole is at disk center
    t1 = time.time()
    dt2 = np.load(f"{gvar.outdir}residual_{gvar.year}_{gvar.day:03d}.npy")
    ph2 = np.load(f"{gvar.outdir}ph_{gvar.year}_{gvar.day:03d}.npy")
    th2 = np.load(f"{gvar.outdir}th_{gvar.year}_{gvar.day:03d}.npy")
    t2 = time.time()
    print(f"Time taken for loading files = {(t2-t1)/60:5.2f} min")

#    radial = fits.open(working_dir + "radialGrid.fits",
#                       memmemmap=False)[0].data.flatten()
#    apod = np.exp( - radial**2/2/0.5**2)
#    apodize = 1/( 1 + np.exp(75*(radial - 0.95)))
    apdz = np.ones_like(dt2)
    dt2mask = ~np.isnan(dt2)

    # masking data, theta, phi
    ph2 = ph2[dt2mask]
    th2 = th2[dt2mask]
    dt2 = dt2[dt2mask]
    apdz = apdz[dt2mask]

    th2 *= np.pi/180.
    ph2 *= np.pi/180
    ph2 += np.pi/2

    # creating healPy maps (including apodization)
    t1 = time.time()
    dt2map, mask2map, th2map, ph2map = make_map(th2, ph2, dt2*apdz, NSIDE)
    t2 = time.time()
    print(f"Time taken for making maps = {(t2-t1)/60:5.2f} min")
    # all coordinates and maps have been converted to healPy maps
    # deleting the temporary variables
    del apdz, th2, ph2, dt2mask

    alm = hp.map2alm(dt2map)
    ellmax = hp.sphtfunc.Alm.getlmax(len(alm))
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(ellmax)

    # alm*2 to account for negative m 
    powspec = computePS(ellArr, emmArr, elllmax, alm*2)

    plt.figure(figsize=(5, 5))
    plt.loglog(powspec)
    plt.xlabel("Spherical harmonic degree $l$")
    plt.ylabel("Velocity in m/s")
    plt.title(" $\sqrt{l \sum_m |A_{lm}|^2}$")
    plt.show()

