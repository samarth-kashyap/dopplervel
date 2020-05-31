# {{{ Library imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sunpy.map
from math import pi
import astropy.units as u
from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a
# }}} imports


# {{{ def loadfits_compressed(fname):
def loadfits_compressed(fname):
    """Loads compressed fits files

    Inputs:
    -------
    fname - str
        filename of the fits file to be loaded

    Outputs:
    --------
    fits_data - np.ndarray(ndim=2, dtype=float)
        the uncompressed image data

    """
    print(f"Reading {fname}")
    fits_file = fits.open(fname)
    fits_file.verify("fix")
    fits_data = fits_file[1].data
    fits_file.close()
    return fits_data
# }}} loadfits_compressed(fname)


# {{{ def plot_los(los1, los2):
def plot_los(los1, los2):
    """Plots images of LOS magnetogram and hat{l} . {ME}

    """
    los_diff = los1 - los2
    fig = plt.figure(figsize=(7, 7))
    plt.subplot(221)
    im = plt.imshow(los1, cmap='seismic')
    plt.colorbar(im)
    plt.title("LOS Magnetogram")

    plt.subplot(222)
    im = plt.imshow(los2, cmap='seismic')
    plt.colorbar(im)
    plt.title("LOS from ME inversion")

    plt.subplot(223)
    im = plt.imshow(los_diff, cmap='seismic')
    plt.colorbar(im)
    plt.title("difference")
    plt.tight_layout()
    return fig
# }}} plot_los(los1, los2)


# {{{ def get_vec_cartesian(b, inc, azi):
def get_vec_cartesian(b, inc, azi):
    """Returns the cartesian components of B using HMI magnetograms

    Inputs:
    -------
    b - np.ndarray(ndim=2, dtype=float)
        |B| i.e. magnitude of the magnetic field
    inc - np.ndarray(ndim=2, dtype=float)
        theta i.e. inclination of the vec{B} wrt los
    azi - np.ndarray(ndim=2, dtype=float)
        phi i.e. azimuthal angle

    Returns:
    --------
    bx, by, bz - np.ndarray(ndim=2, dtype=float)
        x, y, z components of magnetic field

    """
    bz = b * np.cos(inc)
    bx = b * np.sin(inc) * np.cos(azi)
    by = b * np.sin(inc) * np.sin(azi)
    return bx, by, bz
# }}} get_vec_cartesian(b, inc, azi)


# {{{ def create_rot_mat(theta, phi):
def create_rot_mat(theta, phi):
    """Creates a rotation matrix for all data points

    Inputs:
    -------
    theta - np.ndarray(ndim=2, dtype=float)
        2d array containing theta values
    phi - np.ndarray(ndim=2, dtype=float)
        2d array containing phi values

    Returns:
    --------
    rotmat - np.ndarray(ndim=3, dtype=float)
        the set of all rotation matrices

    Notes:
    ------
    The rotation matrix has 3 indices
    [x, y, z] - x corresponds to pixel number
                y, z correspond to the rotation matrix index (0, 1, 2)
    """
    masknan = ~np.isnan(theta)
    theta = theta[masknan].copy()
    phi = phi[masknan].copy()
    rotmat = np.zeros((len(phi), 3, 3), dtype=float)
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)

    rotmat[:, 0, 0] = st * cp
    rotmat[:, 0, 1] = ct * cp
    rotmat[:, 0, 2] = -sp
    rotmat[:, 1, 0] = st * sp
    rotmat[:, 1, 1] = ct * sp
    rotmat[:, 1, 2] = cp
    rotmat[:, 2, 0] = ct
    rotmat[:, 2, 1] = -st
    rotmat[:, 2, 2] = 0

    return rotmat
# }}} create_rot_mat(theta, phi)


# {{{ def create_bxbybz(theta):
def create_bxbybz(theta):
    """Creates a random set of cartesian vectors

    Inputs:
    -------
    theta - np.ndarray(ndim=2, dtype=float)
        array containing set of coordiantes

    Outputs:
    --------
    bx, by, bz
    """
    image_size = theta.shape[0]
    bx = np.random.rand(image_size, image_size)
    by = np.random.rand(image_size, image_size)
    bz = np.random.rand(image_size, image_size)
    masknan = np.isnan(theta)
    bx[masknan] = np.nan
    by[masknan] = np.nan
    bz[masknan] = np.nan
    return bx, by, bz
# }}} create_bxbybz(theta)


# {{{ def get_sph_from_cart(bx, by, bz, rotmat):
def get_sph_from_cart(bx, by, bz, rotmat):
    br = np.zeros(bx.shape)
    bt = np.zeros(bx.shape)
    bp = np.zeros(bx.shape)
    masknan = ~np.isnan(bx)
    br[~masknan] = np.nan
    bt[~masknan] = np.nan
    bp[~masknan] = np.nan

    bx_flat = bx[masknan]
    by_flat = by[masknan]
    bz_flat = bz[masknan]

    br_flat = br[masknan]
    bt_flat = br[masknan]
    bp_flat = br[masknan]

    for i in range(len(bz_flat)):
        bvec = np.array([bx_flat[i], by_flat[i], bz_flat[i]])
        (br_flat[i], bt_flat[i], bp_flat[i]) = bvec.dot(rotmat[i, :, :])

    br[masknan] = br_flat
    bt[masknan] = bt_flat
    bp[masknan] = bp_flat
    return br, bt, bp
# }}} get_sph_from_cart(bx, by, bz, rotmat)


# {{{ def get_vec_polar(b, inc, azi):
def get_vec_polar(b, inc, azi, theta, phi):
    """Returns the cartesian components of B using HMI magnetograms

    Inputs:
    -------
    b - np.ndarray(ndim=2, dtype=float)
        |B| i.e. magnitude of the magnetic field
    inc - np.ndarray(ndim=2, dtype=float)
        theta i.e. inclination of the vec{B} wrt los
    azi - np.ndarray(ndim=2, dtype=float)
        phi i.e. azimuthal angle of vec{B} wrt los
    theta - np.ndarray(ndim=2, dtype=float)
        theta coordiante of pixel
    phi - np.ndarray(ndim=2, dtype=float)
        phi coordiante of pixel

    Returns:
    --------
    br, bt, bp - np.ndarray(ndim=2, dtype=float)
        (r, theta, phi) components of magnetic field

    """
    bx, by, bz = get_vec_cartesian(b, inc, azi)
    rotmat = create_rot_mat(theta, phi)
    br, bt, bp = get_sph_from_cart(bx, by, bz, rotmat)
    return br, bt, bp
# }}} get_vec_polar(b, inc, azi)


# {{{ def get_map_coords(hmi_map):
def get_map_coords(hmi_map):
    # Getting the B0 and P0 angles
    # B0 = hmi_map.observer_coordinate.lat
    # P0 = hmi_map.observer_coordinate.lon
    # r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2)\
    #    / hmi_map.rsun_obs
    # removing all data beyond rcrop (heliocentric radius)
    # rcrop = 0.95
    # mask_r = r > rcrop
    # hmi_map.data[mask_r] = np.nan
    # r[mask_r] = np.nan

    # getting the coordinate data
    x, y = np.meshgrid(*[np.arange(v.value) for v in hmi_map.dimensions])\
        * u.pix
    hpc_coords = hmi_map.pixel_to_world(x, y)
    hpc_hc = hpc_coords.transform_to(frames.Heliocentric)

    # converting from heliocentric cartesian to heliocentric polar
    x = hpc_hc.x.copy()
    y = hpc_hc.y.copy()
    rho = np.sqrt(x**2 + y**2)
    psi = np.arctan2(y, x)

    # heliocentric polar to spherical coordinates (pole @ disk center)
    phi = np.zeros(psi.shape) * u.rad
    phi[psi < 0] = psi[psi < 0] + 2*pi * u.rad
    phi[~(psi < 0)] = psi[~(psi < 0)]
    theta = np.arcsin(rho/hmi_map.rsun_meters)

    return theta, phi
# }}} get_map_coords(hmi_map)


# {{{ def get_jsoc_map(start_time, end_time, series_name, email_notify):
def get_jsoc_map(start_time, end_time, series_name, email_notify):
    """Submits data request on jsoc server and downloads fits files.
    Processes the files and give sunPy map

    Inputs:
    -------
    start_time - str
        Starting time of data in format "yyyy/mm/dd hh:mm:ss"
    end_time - str
        Ending time of data in format "yyyy/mm/dd hh:mm:ss"
    series_name - str
        Name of jsoc series e.g "hmi.M_720_s"
    email_notify - str
        Email address registered in JSOC which will be used to notify

    Returns:
    --------
    jsoc_map - sunPy map
        SunPy map of the given data


    Notes:
    ------
    result = Fido.search(a.Time('2020/05/01 00:00:00', '2020/05/01 00:00:15'),
                         a.jsoc.Series("hmi.M_720s"),
                         a.jsoc.Notify("g.samarth@tifr.res.in"))
    """
    result = Fido.search(a.Time(start_time, end_time),
                         a.jsoc.Series(series_name),
                         a.jsoc.Notify(email_notify))
    print(result)
    downloaded_file = Fido.fetch(result[0])
    print(downloaded_file)
    jsoc_map = sunpy.map.Map(downloaded_file[0])
    return jsoc_map
# }}} get_jsoc_map(start_time, end_time, series_name, email_notify)


# {{{ def plot_b_components(b1, b2, b3, plot_title):
def plot_b_components(b1, b2, b3, plot_title):
    """Plot the individual components and overall magnitude

    Inputs:
    -------
    b1 - np.ndarray(ndim=2, dtype=float)
        component 1 of magnetic field
    b2 - np.ndarray(ndim=2, dtype=float)
        component 2 of magnetic field
    b3 - np.ndarray(ndim=2, dtype=float)
        component 3 of magnetic field
    plot_title - str
        the title of the plot

    Returns:
    --------
    fig - matplotlib figure containing 4 images
        b1, b2, b3 and total magnitude b
    """
    bmag = np.sqrt(b1**2 + b2**2 + b3**2)
    fig = plt.figure()
    plt.title(plot_title)
    plt.subplot(221)
    plt.title("total magnitude")
    im = plt.imshow(bmag)
    plt.colorbar(im)

    plt.subplot(222)
    plt.title("component-1")
    im = plt.imshow(b1, cmap="seismic")
    plt.colorbar(im)

    plt.subplot(223)
    plt.title("component-2")
    im = plt.imshow(b2, cmap="seismic")
    plt.colorbar(im)

    plt.subplot(224)
    plt.title("component-3")
    im = plt.imshow(b3, cmap="seismic")
    plt.colorbar(im)
    plt.tight_layout()

    return fig
# }}} plot_b_components(b1, b2, b3, plot_title)


# {{{ def find_avg(b):
def find_avg(b):
    """Find average of image

    Inputs:
    -------
    b - np.ndarray(ndim=2, dtype=float)
        image

    Outputs:
    bavg - float
        average of image
    """
    masknan = ~np.isnan(b)
    bavg = np.average(b[masknan])
    return bavg
# }}} find_avg(b)


if __name__ == "__main__":
    # {{{ loading files
    data_dir = "/scratch/seismogroup/data/HMI/"
    vec_prefix = "ME_720s_fd10/hmi.ME_720s_fd10.20200501_000000_TAI."
    los_prefix = "M_720s/hmi.M_720s.20200501_000000_TAI.3."
    # b_los = loadfits_compressed(data_dir + los_prefix + "magnetogram.fits")
    # }}} loading files

    b_magnitude = loadfits_compressed(data_dir + vec_prefix + "field.fits")
    b_azimuth = loadfits_compressed(data_dir + vec_prefix + "azimuth.fits")
    b_inclination = loadfits_compressed(data_dir + vec_prefix +
                                        "inclination.fits")
    hmi_map = sunpy.map.Map(data_dir + vec_prefix + "field.fits")

    bx, by, bz = get_vec_cartesian(b_magnitude, b_inclination, b_azimuth)
    theta, phi = get_map_coords(hmi_map)
    rot_mat = create_rot_mat(theta, phi)
    br, bt, bp = get_sph_from_cart(bx, by, bz, rot_mat)
