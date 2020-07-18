# {{{ Library imports
from scipy.spatial.transform import Rotation as R
from globalvars import DopplerVars
from math import sqrt
import healpy as hp
import numpy as np
import argparse
import time
# }}} imports


# {{{ Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on hpc",
                    action="store_true")
parser.add_argument('--cchpc', help="Run program on cchpc19",
                    action="store_true")
parser.add_argument('--job', help="Submit as a job (PBS)",
                    action="store_true")
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


# {{{ def rotate_map_spin_eul(hmap, eulAngle):
def rotate_map_spin_eul(hmap, eulAngle):
    """Take hmap (a healpix map array) and return another healpix map array
    which is ordered such that it has been rotated in (theta, phi) by the
    amounts given.

    Parameters:
    -----------
    hmap - healpy map
        healpy map object

    eulAngle - euler angle

    Returns:
    --------
    rot_map0, rot_map1
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
    rot_map0 = (vth*vph_rot).sum(axis=0) * rot_map0temp +\
        (vph*vph_rot).sum(axis=0) * rot_map1temp

    rot_map1 = (vth*vth_rot).sum(axis=0) * rot_map0temp +\
        (vph*vth_rot).sum(axis=0) * rot_map1temp

    return rot_map0, rot_map1
# }}} rotate_map_spin_eul(hmap, eulAngle)


# {{{ def get_alm_ps(mapp):
def get_alm_ps(mapp):
    """Get alms and the power spectrum

    Parameters:
    -----------
    mapp - healPy map

    Returns:
    --------
    alm - np.ndarray(ndim=1, dtype=complex)
        The spherical harmonic coefficients
    powerSpec - np.ndarray(ndim=1, dtype=float)
        Power spectrum ( sum_{m} |alm|^2 )
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing the ell values
    """
    alm = hp.sphtfunc.map2alm(mapp)
    powerSpec = hp.sphtfunc.anafast(mapp)
    ellArr = np.arange(len(powerSpec))
    return alm, powerSpec, ellArr
# }}} get_alm_ps(mapp)


# {{{ def restructure(ellArr, emmArr, lmax, coefs):
def restructure(ellArr, emmArr, lmax, coefs):
    """Resturcture the alms in the convention followed by pyshtools
    ---------------------------------------------------------------

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
    new_coefs - np.ndarray(ndim=1, dtype=complex)
        The alm array in the new format

    Notes:
    ------
    The convention followed by pyshtools:
    The alm for a given value of ell and emm, the index of the array
    is given by int(ell*(ell + 1)/2 + emm)
    ---------------------------------------------------------------
    """
    assert len(ellArr) == len(emmArr) == len(coefs)
    count = 0
    new_coefs = np.zeros(coefs.shape[0], dtype=complex)
    for ell in range(lmax):
        for emm in range(ell+1):
            index = np.where((ellArr == ell) * (emmArr == emm))[0][0]
            new_coefs[count] = coefs[index]
            count += 1
    return new_coefs
# }}} restructure(ellArr, emmArr, lmax, coefs)


# {{{ def get_only_t(ellArr, emmArr, alm, lmax, t):
def get_only_t(ellArr, emmArr, alm, lmax, t):
    """Filter out the values of alm for a given value of emm

    Parameters:
    -----------
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing the spherical harmonic degree
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing the azimuthal order
    alm - np.ndarray(ndim=1, dtype=complex)
        array containing the spherical harmonic coefficients (alm)
    lmax - int
        maximum value of ell upto which coefficients exists
    t - int
        the azimuthal order to be filtered out

    Returns:
    --------
    alm_new - np.ndarray(ndim=1, dtype=complex)

    """
    count = 0
    alm_new = np.zeros(int(lmax - t + 1), dtype=complex)
    for ell in range(t, lmax+1):
        index = np.where((ellArr == ell) * (emmArr == t))[0][0]
        alm_new[count] = alm[index]
        count += 1
    return alm_new
# }}} get_only_t(ellArr, emmArr, alm, lmax, t)


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


# {{{ def tracking(alm, emmArr, ellArr, lmax, trate, time):
def tracking(alm, emmArr, ellArr, lmax, trate, time):
    """Tracks the data with a given tracking rate.

    Parameters:
    -----------
    alm - np.ndarray(ndim=1, dtype=complex)
        the spherical harmonic coefficients (alm)
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing the emm values
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing the ell values
    lmax - int
        maximum value of ell
    trate - float
        tracking rate in Hz
    time - float
        time in seconds

    Returns:
    --------
    alm2 - np.ndarray(ndim=1, dtype=complex)
        Spherical harmonic coefficients after tracking

    Notes:
    ------
    By tracking spherical harmonic coefficients, we effectively introduce
    an additional phase of e^{i t T} where T is the tracking rate and t is
    the total time.

    """
    alm2 = alm.copy()
    for emm in range(lmax+1):
        index = np.where((emmArr == emm))[0]
        tomega = 1j * emm * trate
        alm2[index] *= np.exp(tomega)
    return alm2
# }}} tracking(alm, emmArr, ellArr, lmax, trate, time)


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
    alm_v = -alm_pm[0]
    alm_w = -1j*alm_pm[1]
    return alm_r, alm_v, alm_w
# }}} get_spin1_alms(map_r, map_trans)


# {{{ def get_spin1_maps(dat_map, msk_map, th_map, ph_map, pole="diskCenter"):
def get_spin1_maps(data_map, mask_map, theta_map, phi_map, pole="diskCenter"):
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
    if pole == "diskCenter":
        losr = np.cos(theta_map)
        lost = -np.sin(theta_map)
        losp = 0 * lost
    elif pole == "solarNorth":
        losr = np.sin(theta_map) * np.cos(phi_map)
        lost = - np.cos(theta_map) * np.cos(phi_map)
        losp = np.sin(phi_map)
    else:
        print("unrecognized coordinate system")
        exit()

    # Finding the vector spherical harmonic coefficients
    map_r = data_map * losr
    map_p = - data_map * (lost + 1j*losp) / sqrt(2)
    map_m = - data_map * (lost - 1j*losp) / sqrt(2)
    map_trans = [(map_p + map_m)/2, -1j*(map_p - map_m)/2]

    map_r[~mask_map] = hp.UNSEEN
    map_trans[0][~mask_map] = 0.0
    map_trans[1][~mask_map] = 0.0

    return map_r, map_trans
# }}} get_spin1_maps(data_map, mask_map, theta_map, phi_map, pole="diskCenter")


if __name__ == "__main__":
    rw_dir = gvar.outdir
    # suffix convention
    # 1 - used for coordinates where pole is at solar north pole
    # 2 - used for coordinates where pole is at disk center
    t1 = time.time()
    dt2 = np.load(f"{gvar.outdir}residual_{gvar.year}_{gvar.day:03d}.npy")
    ph2 = np.load(f"{gvar.outdir}phDC_{gvar.year}_{gvar.day:03d}.npy")
    th2 = np.load(f"{gvar.outdir}thDC_{gvar.year}_{gvar.day:03d}.npy")
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

    # creating healPy maps (including apodization)
    t1 = time.time()
    dt2map, mask2map, th2map, ph2map = make_map(th2, ph2, dt2*apdz, NSIDE)
    t2 = time.time()
    print(f"Time taken for making maps = {(t2-t1)/60:5.2f} min")
    # all coordinates and maps have been converted to healPy maps
    # deleting the temporary variables
    del apdz, th2, ph2, dt2mask

    map2r, map2trans = get_spin1_maps(dt2map, mask2map,
                                      th2map, ph2map, pole="diskCenter")
    alm2r, alm2v, alm2w = get_spin1_alms(map2r, map2trans)
    ellmax = hp.sphtfunc.Alm.getlmax(len(alm2r))
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(ellmax)

    np.save(f"{rw_dir}ulm{gvar.day:03d}.npy", alm2r)
    np.save(f"{rw_dir}vlm{gvar.day:03d}.npy", alm2v)
    np.save(f"{rw_dir}wlm{gvar.day:03d}.npy", alm2w)
#    np.savez_compressed(data_dir + "ulm"+str(days).zfill(3)+".txt.npz",
#                        ulm=alm2r)
#    np.savez_compressed(data_dir + "vlm"+str(days).zfill(3)+".txt.npz",
#                        vlm=alm2v)
#    np.savez_compressed(data_dir + "wlm"+str(days).zfill(3)+".txt.npz",
#                        wlm=alm2w)
#    np.savez_compressed(data_dir + "ellArr.txt.npz", ellArr=ellArr)
#    np.savez_compressed(data_dir + "emmArr.txt.npz", emmArr=emmArr)

    # Rotation of maps needs to be done after inversion
    # map1trans = rotate_map_spin_eul(map_trans, np.array([0, -pi/2, 0]))
