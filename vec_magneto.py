import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


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
    print(f"Loading {fname} ...")
    fits_file = fits.open(fname)
    fits_file.verify("fix")
    fits_data = fits_file[1].data
    fits_file.close()
    print(f"Loading {fname} complete")
    return fits_data


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


if __name__ == "__main__":
    data_dir = "/scratch/seismogroup/data/HMI/"
    vec_prefix = "ME_720s_fd10/hmi.ME_720s_fd10.20200501_000000_TAI."
    los_prefix = "M_720s/hmi.M_720s.20200501_000000_TAI.3."
    b_magnitude = loadfits_compressed(data_dir + vec_prefix + "field.fits")
    b_azimuth = loadfits_compressed(data_dir + vec_prefix + "azimuth.fits")
    b_inclination = loadfits_compressed(data_dir + vec_prefix +
                                        "inclination.fits")
    b_los = loadfits_compressed(data_dir + los_prefix + "magnetogram.fits")

    b_los_from_vec = b_magnitude * np.cos(b_inclination)
    fig = plot_los(b_los, b_los_from_vec)
    plt.show()
