# {{{ Library imports
from pyshtools import legendre as pleg
from sunpy.coordinates import frames
#from heliosPy import iofuncs as cio
from sunpy.map import Map as spMap
from astropy.io import fits
import astropy.units as u
import pickle as pkl
from math import pi
import numpy as np
import argparse
import time
import os
# }}} imports


# {{{ argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on cluster",
                    action="store_true")
parser.add_argument('--job', help="Submit as a job (PBS)",
                    action="store_true")
parser.add_argument('--gnup', help="Argument for gnuParallel",
                    type=int)
args = parser.parse_args()

if args.hpc:
    scratch_dir = "/scratch/g.samarth/"
else:
    scratch_dir = "/home/samarthgk/hpcscratch/"
# }}} argument parser


# {{{ def get_pleg_index(l, m):
def get_pleg_index(l, m):
    """Gets the index for accessing legendre polynomials
    (generated from pyshtools.legendre)

    Parameters:
    -----------
    l : int
        Spherical Harmonic degree
    m : int
        Azimuthal order

    Returns:
    --------
    int
        index for accessing legendre polynomial
    """
    return int(l*(l+1)/2 + m)
# }}} get_pleg_index(l, m):


# {{{ def gen_leg(lmax, theta):
def gen_leg(lmax, theta):
    """Generates associated legendre polynomials and derivatives

    Parameters:
    -----------
    lmax : int
        Maximum spherical harmonic degree
    theta : np.ndarray(ndim=1, dtype=np.float64)
        1D array containing theta for computing P_l(cos(theta))

    Returns:
    --------
    (leg, leg_d1) : list of np.ndarray(ndim=2)
        Legendre polynomials and it's derivatives
    """
    cost = np.cos(theta)
    sint = np.sin(theta)
    maxIndex = int(lmax+1)
#    ell = np.arange(lmax+1)
    leg = np.zeros((maxIndex, theta.size))
    leg_d1 = np.zeros((maxIndex, theta.size))

    count = 0
    for z in cost:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2), \
        leg_d1 * (-sint).reshape(1, sint.shape[0])/np.sqrt(2)
# }}} gen_leg(lmax, theta)


# {{{ def gen_leg_x(lmax, x):
def gen_leg_x(lmax, x):
    """Generates associated legendre polynomials and derivatives
    for a given x

    Parameters:
    -----------
    lmax : int
        Maximum spherical harmonic degree
    x: float
       x for computing P_l(x)

    Returns:
    --------
    (leg, leg_d1) : list
        Legendre polynomial and it's derivative
    """
    maxIndex = int(lmax+1)
#    ell = np.arange(lmax+1)
    leg = np.zeros((maxIndex, x.size))
    leg_d1 = np.zeros((maxIndex, x.size))

    count = 0
    for z in x:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2), leg_d1/np.sqrt(2)
# }}} gen_leg_x(lmax, x)


# {{{ def smooth(img):
def smooth(img):
    """Smoothen by averaging over neighboring pixels.

    Parameters:
    -----------
    img : np.ndarray(ndim=2, dtype=np.float64)
        Input image

    Returns:
    --------
    avg_img : np.ndarray(ndim=2, dtype=np.float64)
        Smoothened image
    """
    avg_img = (img[1:-1, 1:-1] +  # center
               img[:-2, 1:-1] +   # top
               img[2:, 1:-1] +    # bottom
               img[1:-1, :-2] +   # left
               img[1:-1, 2:]) / 5.0     # right
    return avg_img
# }}} smooth(img)


# {{{ def downsample(img, N):
def downsample(img, N):
    for i in range(N):
        img = smooth(img)
    return img
# }}} downsample(img, N)


# {{{ def inv_SVD(A, svdlim):
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
    sinv[sinv/sinv[0] > svdlim] = 0.0  # svdlim
    return np.dot(v.transpose().conjugate(),
                  np.dot(np.diag(sinv), u.transpose().conjugate()))
# }}} inv_SVD(A, svdlim)


# {{{ def inv_reg1(A, regparam):
def inv_reg1(A, regparam):
    Ashape = A.shape[0]
    return np.linalg.inv(A.transpose().conjugate().dot(A) +
                         regparam *
                         np.identity(Ashape)).dot(A.transpose().conjugate())
# }}} inv_reg1(A, regparam)


# {{{ def inv_reg2(A, regparam):
def inv_reg2(A, regparam):
    reg2 = 2*np.identity(A.shape[0])
    offd2 = -1*np.identity(A.shape[0]-1)
    reg2[1:, :-1] += offd2
    reg2[:-1, 1:] += offd2
    reg = reg2[1:-1, :].copy()
    return np.linalg.inv(A.transpose().dot(A) +
                         (regparam/16.) *
                         reg.transpose().dot(reg)).dot(A.transpose())
# }}} inv_reg2(A, regparam)


# {{{ def inv_reg3(A, regparam):
def inv_reg3(A, regparam):
    reg2 = 3*np.identity(A.shape[0])
    offd2 = -1*np.identity(A.shape[0]-1)
    reg2[:-1, 1:] += 3*offd2
    reg2[1:, :-1] += offd2
    reg2[:-2, 2:] += -offd2[1:, 1:]
    reg = reg2[1:-2, :].copy()
    return np.linalg.inv(A.transpose().dot(A) \
                + (regparam/64.) * reg.transpose().dot(reg))\
                    .dot(A.transpose())
# }}} inv_reg3(A, regparam)


if __name__ == "__main__":
    hmi_data_dir = scratch_dir + "HMIDATA/v720s_dConS/2018/"
    hmi_data_fnames = hmi_data_dir + "HMI_2018_filenames"
    plot_dir = scratch_dir + "plots/dopplervel/"
    print("Program started -- reading files")
    print(f" HMI data fnames = {hmi_data_fnames}")

    with open(hmi_data_fnames, mode="r") as f:
        hmi_files = f.read().splitlines()

    total_days = len(hmi_files)
    print(f"Total days = {total_days}")

    if args.job:
        try:
            procid = int(os.environ['PBS_VNODENUM'])
        except KeyError:
            pass
        nproc = 6
        print(f" procid = {procid}")
        daylist = np.arange(procid, total_days, nproc)
    elif args.gnup:
        daylist = np.array([args.gnup])
        if args.gnup>total_days:
            print(f" Invalid argument for --gnup {args.gnup} . Must be " +
                  f" less than {total_days}")
            exit()
    else:
        daylist = np.arange(0, 2)

    for day in daylist:
        # loading HMI image as sunPy map
        hmi_map = spMap(hmi_data_dir + hmi_files[day])

        # Getting the B0 and P0 angles
        B0 = hmi_map.observer_coordinate.lat
        P0 = hmi_map.observer_coordinate.lon

        # setting up for data cleaning
        x, y = np.meshgrid(*[np.arange(v.value) for v in hmi_map.dimensions])\
            * u.pix
        hpc_coords = hmi_map.pixel_to_world(x, y)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2)\
            / hmi_map.rsun_obs

        # removing all data beyond rcrop (heliocentric radius)
        rcrop = 0.95
        mask_r = r > rcrop

        hmi_map.data[mask_r] = np.nan
        r[mask_r] = np.nan

        hpc_hgf = hpc_coords.transform_to(frames.HeliographicStonyhurst)
        hpc_hc = hpc_coords.transform_to(frames.Heliocentric)

        x = hpc_hc.x.copy()
        y = hpc_hc.y.copy()
        z = hpc_hc.z.copy()
        rho = np.sqrt(x**2 + y**2)
        psi = np.arctan2(y, x)

        phi = np.zeros(psi.shape) * u.rad
        phi[psi<0] = psi[psi<0] + 2*pi * u.rad
        phi[~(psi<0)] = psi[~(psi<0)]
        theta = np.arcsin(rho/hmi_map.rsun_meters)

        map_data = hmi_map.data.copy()
        mask_nan = ~np.isnan(map_data)
        # -- ( plot ) determining maximum and minimum pixel values --
        """
        # -----------------------------------------------------------
        _max_data = map_data[mask_nan].max()
        _min_data = map_data[mask_nan].min()
        maxx = max(abs(_max_data), abs(_min_data))
        print(_max_data, _min_data)
        del map_data, _max_data, _min_data
        plt.figure()
        im = plt.imshow(hmi_map.data, cmap='seismic', interpolation="none")
        plt.colorbar(im)
        plt.axis("off")
        plt.savefig(plot_dir + "rawMap.png", dpi=500)
        plt.show()
        # -----------------------------------------------------------
        """

        # Removing effect of satellite velocity
        map_fits = fits.open(hmi_data_dir + hmi_files[0])
        map_fits.verify('fix')
        vx = map_fits[1].header['OBS_VR']
        vz = map_fits[1].header['OBS_VN']
        vy = map_fits[1].header['OBS_VW']
        VR, VN, VW = vx, vz, vy
        print(f"VR = {VR}, VN = {VN}, VW = {VW}")
        del map_fits

        x = hpc_hc.x
        y = hpc_hc.y
        sChi = np.sqrt(x**2 + y**2)
        sSig = np.sin(np.arctan(r/200))
        cSig = np.cos(np.arctan(r/200))
        chi = np.arctan2(x, y)
        sChi = np.sin(chi)
        cChi = np.cos(chi)

        thetaHGC = hpc_hgf.lat.copy()  # + 90*u.deg
        phiHGC = hpc_hgf.lon.copy()
        thHG1 = thetaHGC[mask_nan]
        phHG1 = phiHGC[mask_nan]
        del thetaHGC, phiHGC

        ct = np.cos(thHG1)
        st = np.sin(thHG1)
        cp = np.cos(phHG1)
        sp = np.sin(phHG1)

        vr1 = cp*st*vx + sp*st*vy + ct*vz
        vt1 = cp*ct*vx + sp*ct*vy - st*vz
        vp1 = -sp*vx + cp*vy

        sB0 = np.sin(B0)
        cB0 = np.cos(B0)

        lr = sB0*ct + cB0*st*cp
        lt = sB0*st - cB0*ct*cp
        lp = cB0*sp

        vC1 = lr*vr1 + lt*vt1 + lp*vp1
        velCorr = np.zeros((4096, 4096))
        velCorr[mask_nan] = vC1
        velCorr[~mask_nan] = np.nan

        # method 2
        vr1 = VR*cSig
        vr2 = -VW*sSig*sChi
        vr3 = -VN*sSig*cChi
        vC1 = vr1 + vr2 + vr3
        velCorr = np.zeros((4096, 4096))
        velCorr[mask_nan] = vC1[mask_nan] + 632
        velCorr[~mask_nan] = np.nan

        tempFull = np.zeros((4096, 4096))
        tempFull[~mask_nan] = np.nan

        map_data += velCorr 
        map_data -= 632

        lat = hpc_hgf.lat.copy() + 90*u.deg
        lon = hpc_hgf.lon.copy()
        cB0 = np.cos(B0)
        sB0 = np.sin(B0)

        lat1D = lat[mask_nan].copy()
        lon1D = lon[mask_nan].copy()
        rho1D = r[mask_nan].copy()
        ct = np.cos(lat1D)
        st = np.sin(lat1D)
        sp = np.sin(lon1D)
        cp = np.cos(lon1D)

        t1 = time.time()
        pl_theta, dt_pl_theta = gen_leg(5, lat1D)
        ell = np.arange(6)
        t2 = time.time()
        print(f"Time taken for pl_theta = {(t2 - t1)/60} minutes")
        pl_rho, dt_pl_rho = gen_leg_x(5, rho1D)
        t3 = time.time()
        print(f"Time taken for pl_rho = {(t3 - t2)/60} minutes")

        # Normalizing the functions
        dt_pl_theta /= np.sqrt(ell*(ell+1)).reshape(6, 1)
        dt_pl_rho /= np.sqrt(ell*(ell+1)).reshape(6, 1)

        lt = sB0 * st - cB0 * ct * cp
        lp = cB0 * sp
        imArr = np.zeros((11, lt.shape[0]))
        A = np.zeros((11, 11))
        imArr[0, :] = dt_pl_theta[1, :] * lp
        imArr[1, :] = dt_pl_theta[3, :] * lp
        imArr[2, :] = dt_pl_theta[5, :] * lp

        #imArr[3, :] = dt_pl_theta[0, :] * lt
        imArr[3, :] = dt_pl_theta[2, :] * lt
        imArr[4, :] = dt_pl_theta[4, :] * lt

        imArr[5, :] = pl_rho[0, :]
        imArr[6, :] = pl_rho[1, :]
        imArr[7, :] = pl_rho[2, :]
        imArr[8, :] = pl_rho[3, :]
        imArr[9, :] = pl_rho[4, :]
        imArr[10, :] = pl_rho[5, :]

        mapArr = map_data[mask_nan].copy()

        RHS = imArr.dot(mapArr)

        for i in range(11):
            for j in range(11):
                A[i, j] = imArr[i, :].dot(imArr[j, :])

        Ainv = inv_SVD(A, 1e5)
        fitParams = Ainv.dot(RHS)
        print(f" ##(in m/s)## Rotation = {fitParams[:3]},"\
              + f"\nMeridional Circ = {fitParams[3:5]},"\
              + f"\nLimb Shift = {fitParams[5:]}\n")
        print(f" ##(in Hz)## Rotation = {fitParams[:3]/2/pi/0.695},"\
              + f"\nMeridional Circ = {fitParams[3:5]/2/pi/0.695},"\
              + f"\nLimb Shift = {fitParams[5:]}")
        newImgArr = fitParams.dot(imArr)
        new1 = fitParams[:3].dot(imArr[:3, :])
        new2 = fitParams[3:5].dot(imArr[3:5, :])
        new3 = fitParams[5:].dot(imArr[5:, :])

        newImg = np.zeros((4096, 4096))
        newImg[mask_nan] = newImgArr
        newImg[~mask_nan] = np.nan
        resImg = map_data - newImg
        """
        cio.writefitsfile(map_data - newImg,
                          hmi_data_dir + "residual"+str(day).zfill(3)+".fits")
        cio.writefitsfile((hpc_hgf.lat + 90*u.deg).value,
                          hmi_data_dir + "theta"+str(day).zfill(3)+".fits")
        cio.writefitsfile(hpc_hgf.lon.value ,
                          hmi_data_dir + "phi"+str(day).zfill(3)+".fits")
        cio.writefitsfile(theta.value,
                          hmi_data_dir + "thetaRot"+str(day).zfill(3)+".fits")
        cio.writefitsfile(phi.value,
                          hmi_data_dir + "phiRot"+str(day).zfill(3)+".fits")
        """
        pkl.dump(map_data - newImg,
                open(hmi_data_dir + "residual"+str(day).zfill(3)+".pkl","wb"))
        pkl.dump((hpc_hgf.lat + 90*u.deg).value,
                open(hmi_data_dir + "theta"+str(day).zfill(3)+".pkl","wb"))
        pkl.dump(hpc_hgf.lon.value,
                open(hmi_data_dir + "phi"+str(day).zfill(3)+".pkl","wb"))
        pkl.dump(theta.value,
                open(hmi_data_dir + "thetaRot"+str(day).zfill(3)+".pkl","wb"))
        pkl.dump(phi.value, 
                open(hmi_data_dir + "phiRot"+str(day).zfill(3)+".pkl","wb"))
        del map_data, newImg, hpc_hgf, resImg, new1, new2, new3
        del newImgArr, fitParams, Ainv, imArr
        del dt_pl_rho, dt_pl_theta, st, ct, sp, cp
