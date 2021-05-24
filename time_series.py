# {{{ Library imports
import matplotlib.pyplot as plt            # Plotting
from heliosPy import iofuncs as cio
import healpy as hp
import numpy as np
import argparse
# }}} imports


NAX = np.newaxis

# {{{ ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on daahpc",
                    action="store_true")
parser.add_argument('--cchpc', help="Run program on cchpc19",
                    action="store_true")
parser.add_argument('--datatype', help="\'LCT\' or \'doppler\'",
                    type=str)
parser.add_argument('--chris', help="Data from chris",
                    action="store_true")
parser.add_argument('--year', help='year', type=int)
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
# }}} parser

datatype = args.datatype

if datatype == 'doppler':
    data_dir = scratch_dir + "HMIDATA/data_analysis/lmax1535/"
elif datatype == 'LCT':
    if args.chris:
        lctdir = "LCT_chris"
    else:
        lctdir = "LCT"
    data_dir = f"/scratch/g.samarth/HMIDATA/{lctdir}/"

NTIME = 350
LMAXSUM = 25


# {{{ def load_time_series(NTIME, data_dir):
def load_time_series(NTIME, data_dir, datatype='doppler'):
    '''Loads time series of alms by reading alms of each image.

    Parameters:
    -----------
    NTIME - int
        length of the time series
    data_dir - string
        directory in which the alms of the image is located

    Returns:
    --------
    utime = np.ndarray(ndim=2, dtype=complex)
        time series of radial spherical harmonic
    vtime = np.ndarray(ndim=2, dtype=complex)
        time series of poloidal spherical harmonic
    wtime = np.ndarray(ndim=2, dtype=complex)
        time series of toroidal spherical harmonic

    Notes:
    ------
    axis=0 - spherical harmonic indices
    axis=1 - time

    '''
    count = 0
    for i in range(NTIME):
        if datatype == 'doppler':
            suffix = str(i).zfill(3) + ".npz"
            fname = data_dir + "alm.data.inv.final" + suffix
        elif datatype == 'LCT':
            fname = data_dir + f"almo_{args.year}_{i:03d}.npz"
        try:
            alm = np.load(fname)
            alm_found = True
        except FileNotFoundError:
            print(f"{fname} not found")
            alm_found = False
            """
            if count==0:
                alm_found = False
            else:
                alm_found = True
            """
            pass
        if count==0 and alm_found:
            if datatype == 'doppler':
                ulm = alm['ulm']
            vlm = alm['vlm']
            wlm = alm['wlm']
            # lmax = alm['ellmax']

            ulen = len(vlm)
            if datatype == 'doppler':
                utime = np.zeros((ulen, NTIME), dtype=complex)
            vtime = np.zeros((ulen, NTIME), dtype=complex)
            wtime = np.zeros((ulen, NTIME), dtype=complex)

            if datatype == 'doppler':
                utime[:, 0] = ulm
            vtime[:, 0] = vlm
            wtime[:, 0] = wlm
            print(f" i = {i} ")
            count += 1
        elif count>0 and alm_found:
            print(f"Loaded {fname}")
            if datatype == 'doppler':
                utime[:, i] = alm['ulm']
            vtime[:, i] = alm['vlm']
            wtime[:, i] = alm['wlm']
            if i%10==0:
                print(f" i = {i} ")
    if datatype == 'doppler':
        return utime, vtime, wtime
    elif datatype == 'LCT':
        return vtime*0.0, vtime, wtime
# }}} load_time_series(NTIME, data_dir)


# {{{ def get_s_plot(ufreq, vfreq, wfreq, ellArr, emmArr, s, t):
def get_s_plot(ufreq, vfreq, wfreq, ellArr, emmArr, s, t):
    '''Takes the spectral coefficients and returns an array for a given s

    Parameters:
    -----------
    ufreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic
    vfreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic
    wfreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic
    ellArr - np.ndarray(ndim=1, dtype=int)
        list of ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        list of emm values
    s - int
        spherical harmonic degree
    t - int
        azimuthal order

    Returns:
    --------
    ufreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic for s, t
    vfreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic for s, t
    wfreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic for s, t

    '''
    len_alm = ufreq.shape[0]
    lmax = hp.sphtfunc.Alm.getlmax(len_alm)
    assert (s <= lmax) and (t <= s)
    isess = ellArr == s
    istee = emmArr == t
    mask = isess * istee
    ufreq_st = ufreq[mask, :]
    vfreq_st = vfreq[mask, :]
    wfreq_st = wfreq[mask, :]

    return ufreq_st, vfreq_st, wfreq_st
# }}} get_s_plot(ufreq, vfreq, wfreq, ellArr, emmArr, s, t)


# {{{ def get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, lmin, lmax):
def get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, lmin, lmax):
    '''Takes the spectral coefficients and returns an array for a given s

    Parameters:
    -----------
    ufreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic
    vfreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic
    wfreq - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic
    ellArr - np.ndarray(ndim=1, dtype=int)
        list of ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        list of emm values
    lmax - int
        maximum value of ell upto which power is summed

    Returns:
    --------
    ufreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic for s, t
    vfreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic for s, t
    wfreq_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic for s, t

    '''
    len_alm = ufreq.shape[0]
    len_omega = ufreq.shape[1]
#    lmax = hp.sphtfunc.Alm.getlmax(len_alm)
    ufreq_st = np.zeros((lmax+1, len_omega))
    vfreq_st = np.zeros((lmax+1, len_omega))
    wfreq_st = np.zeros((lmax+1, len_omega))
    """
    for ess in range(lmax+1):
        for tee in range(ess+1):
            st = int(ess - tee)
            isess = ellArr==ess
            istee = emmArr==tee
            mask = isess * istee
            ufreq_st[st, :] += abs(ufreq[mask, :]).flatten()
            vfreq_st[st, :] += abs(vfreq[mask, :]).flatten()
            wfreq_st[st, :] += abs(wfreq[mask, :]).flatten()
    """
    for st in range(lmax+1):
        mask = (ellArr - emmArr == st) * (ellArr < lmax) * (ellArr >= lmin)
        ufreq_st[st, :] += abs(ufreq[mask, :]).sum(axis=0)
        vfreq_st[st, :] += abs(vfreq[mask, :]).sum(axis=0)
        wfreq_st[st, :] += abs(wfreq[mask, :]).sum(axis=0)
    return ufreq_st, vfreq_st, wfreq_st
# }}} get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, lmin, lmax)


# {{{ def plot_all(ust, vst, wst, lmin, lmax):
def plot_all(ust, vst, wst, lmin, lmax):
    xst = np.arange(lmax+1)

    plt.figure(figsize=(10, 10))
#    plt.rcParams.update({'font.size': 20})
    # plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size="20")
    plt.title(" Total power summed over $s_{min}$ =" + f" {lmin}," +"$s_{max}$ = " + f"{lmax}")
    plt.semilogy(xst, ust.sum(axis=1)/NTIME, 'g', label='radial')
    plt.semilogy(xst, vst.sum(axis=1)/NTIME, 'r', label='poloidal')
    plt.semilogy(xst, wst.sum(axis=1)/NTIME, 'b', label='toroidal')
    plt.xlabel("$s - |t|$")
    plt.ylabel("Total Power in ms$^{-1}$")
    plt.legend()
    plot_fname = data_dir + "st_log_"+str(lmin)+"_"+str(lmax)+".pdf"
    print(f"Saving plot to {plot_fname}")
    plt.savefig(plot_fname)
    plt.close()

    plt.figure(figsize=(10, 10))
#    plt.rcParams.update({'font.size': 20})
    # plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size="20")
    plt.title(" Total power summed over $s_{min}$ =" + f" {lmin}," +"$s_{max}$ = " + f"{lmax}")
    plt.plot(xst, ust.sum(axis=1)/NTIME, 'g', label='radial')
    plt.plot(xst, vst.sum(axis=1)/NTIME, 'r', label='poloidal')
    plt.plot(xst, wst.sum(axis=1)/NTIME, 'b', label='toroidal')
    plt.xlabel("$s - |t|$")
    plt.ylabel("Total Power in ms$^{-1}$")
    plt.legend()
    plot_fname = data_dir + "st_"+str(lmin)+"_"+str(lmax)+".pdf"
    print(f"Saving plot to {plot_fname}")
    plt.savefig(plot_fname)
    plt.close()
    return None
# }}} plot_all(ust, vst, wst, lmin, lmax)


# {{{ def plot_freq(ust, vst, wst):
def plot_freq(ust, vst, wst):
    plt.figure(figsize=(30, 20))
    plt.rcParams.update({'font.size': 15})
    plt.subplot(311)
    im = plt.imshow(ust, aspect=11/smax,
                extent=[-5.753,5.753, ust.shape[0], 0])
    plt.xlabel("$\sigma$ in $\mu$Hz")
    plt.ylabel("$s - |t|$")
    plt.title("radial")
    plt.colorbar(im)

    plt.subplot(312)
    im = plt.imshow(vst, aspect=11/smax,
                extent=[-5.753, 5.753, vst.shape[0], 0])
    plt.xlabel("$\sigma$ in $\mu$Hz")
    plt.ylabel("$s - |t|$")
    plt.title("poloidal")
    plt.colorbar(im)

    plt.subplot(313)
    im = plt.imshow(wst, aspect=11/smax, 
                    extent=[-5.753, 5.753, wst.shape[0], 0])
    plt.xlabel("$\sigma$ in $\mu$Hz")
    plt.ylabel("$s - |t|$")
    plt.title("toroidal")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
    return None
# }}} plot_freq(ust, vst, wst)


# {{{ def analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax)
def analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax):
    num_blocks = int(lmax/block_size)
    lmin, lmax = 0, 0
    for i in range(num_blocks-1):
        print(f"Block number = {i+1} of {num_blocks-1}")
        lmin = lmax
        lmax = lmin + block_size
        ust, vst, wst = get_st_plot(utime, vtime, wtime,
                                    ellArr, emmArr, lmin, lmax)
        plot_all(ust, vst, wst, lmin, lmax)
    return None
# }}} analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax)



def compute_lnu(tracking=True):
    lmax = min(ellArr.max(), 1500) 
    upow = np.zeros((lmax, NTIME), dtype=np.complex128)
    vpow = np.zeros((lmax, NTIME), dtype=np.complex128)
    wpow = np.zeros((lmax, NTIME), dtype=np.complex128)
    for ell in range(lmax):
        print(f"ell = {ell}")
        mask_ell = ellArr == ell
        if tracking:
            m = emmArr[mask_ell]
            omega = 453.1*1e-9*2*np.pi
            eimot = np.exp(1j*m[:, NAX]*omega*time_arr[NAX, :])
            utrack = utime[mask_ell, :]*eimot
            vtrack = vtime[mask_ell, :]*eimot
            wtrack = wtime[mask_ell, :]*eimot
            upow[ell, :] = np.sum(abs(np.fft.fft(utrack, axis=1))**2, axis=0)
            vpow[ell, :] = np.sum(abs(np.fft.fft(vtrack, axis=1))**2, axis=0)
            wpow[ell, :] = np.sum(abs(np.fft.fft(wtrack, axis=1))**2, axis=0)
        else:
            upow[ell, :] = np.sum(abs(np.fft.fft(utime[mask_ell, :],
                                                 axis=1))**2, axis=0)
            vpow[ell, :] = np.sum(abs(np.fft.fft(vtime[mask_ell, :],
                                                 axis=1))**2, axis=0)
            wpow[ell, :] = np.sum(abs(np.fft.fft(wtime[mask_ell, :],
                                                 axis=1))**2, axis=0)
        upow[ell, :] *= ell*(ell+1)/NTIME*2
        vpow[ell, :] *= ell*(ell+1)/NTIME*2
        wpow[ell, :] *= ell*(ell+1)/NTIME*2
    return (upow, vpow, wpow)



def compute_pow_prasad(tracking=False):
    lmax = min(ellArr.max(), 1500)
    lmax = 25
    upow = np.zeros(NTIME, dtype=np.float)
    vpow = np.zeros(NTIME, dtype=np.float)
    wpow = np.zeros(NTIME, dtype=np.float)
    for ell in range(1, lmax, 2):
        print(f"ell = {ell}")
        mask_ell = ellArr == ell
        if tracking:
            m = emmArr[mask_ell]
            omega = 453.1*1e-9*2*np.pi
            eimot = np.exp(1j*m[:, NAX]*omega*time_arr[NAX, :])
            utrack = utime[mask_ell, :]*eimot
            vtrack = vtime[mask_ell, :]*eimot
            wtrack = wtime[mask_ell, :]*eimot
            upow[ell, :] = np.sum(abs(np.fft.fft(utrack, axis=1))**2, axis=0)
            vpow[ell, :] = np.sum(abs(np.fft.fft(vtrack, axis=1))**2, axis=0)
            wpow[ell, :] = np.sum(abs(np.fft.fft(wtrack, axis=1))**2, axis=0)
        else:
            upow += np.sum(2*ell*(ell+1)*abs(np.fft.fft(utime[mask_ell, :],
                                                        axis=1))**2/NTIME/NTIME, axis=0)
            vpow += np.sum(2*ell*(ell+1)*abs(np.fft.fft(vtime[mask_ell, :],
                                                        axis=1))**2/NTIME/NTIME, axis=0)
            wpow += np.sum(2*ell*(ell+1)*abs(np.fft.fft(wtime[mask_ell, :],
                                                        axis=1))**2/NTIME/NTIME, axis=0)
    upow = np.sqrt(upow)
    vpow = np.sqrt(vpow)
    wpow = np.sqrt(wpow)
    return (upow, vpow, wpow)


def plot_lnu(uvw):
    upow, vpow, wpow = uvw
    lmin = 20
    maskpos = freq_arr >= 0
    upow = upow[lmin:, maskpos]
    vpow = vpow[lmin:, maskpos]
    wpow = wpow[lmin:, maskpos]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    umax = abs(upow[~np.isnan(upow)]).max()
    vmax = abs(vpow[~np.isnan(vpow)]).max()
    wmax = abs(wpow[~np.isnan(wpow)]).max()
    axs.flatten()[0].imshow(abs(upow), vmax=umax, aspect='auto')
    axs.flatten()[1].imshow(abs(vpow), vmax=vmax, aspect='auto')
    axs.flatten()[2].imshow(abs(wpow), vmax=wmax, aspect='auto')
    return fig, axs


def plot_nu(uvw):
    upow, vpow, wpow = uvw
    ell = np.arange(upow.shape[0])
    lmin = 5
    maskpos = freq_arr >= 0
    # masklmax = ell <= LMAXSUM
    masklmax = np.ones_like(ell, dtype=np.bool)
    upow = np.sqrt(upow[masklmax, :].sum(axis=0))[maskpos]
    vpow = np.sqrt(vpow[masklmax, :].sum(axis=0))[maskpos]
    wpow = np.sqrt(wpow[masklmax, :].sum(axis=0))[maskpos]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    axs.flatten()[0].plot(freq_arr[maskpos], abs(upow))
    axs.flatten()[1].plot(freq_arr[maskpos], abs(vpow))
    axs.flatten()[2].plot(freq_arr[maskpos], abs(wpow))
    return fig, axs


def plot_prasad(uvw):
    upow, vpow, wpow = uvw
    maskpos = freq_arr >= 0
    upow = upow[maskpos]
    vpow = vpow[maskpos]
    wpow = wpow[maskpos]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    axs.flatten()[0].plot(freq_arr[maskpos], abs(upow))
    axs.flatten()[1].plot(freq_arr[maskpos], abs(vpow))
    axs.flatten()[2].plot(freq_arr[maskpos], abs(wpow))
    return fig, axs



if __name__=="__main__":
    if args.chris:
        max_time = NTIME*24*3600  # total time in seconds
    else:
        max_time = NTIME*24*3600  # total time in seconds
    time_arr = np.linspace(0, max_time, NTIME)
    dtime = time_arr[1] - time_arr[0]
    freq_arr = np.fft.fftfreq(time_arr.shape[0], dtime)*1e-6
    dfreq = freq_arr[1] - freq_arr[0]

    utime, vtime, wtime = load_time_series(NTIME, data_dir,
                                           datatype=args.datatype)
    length_alm = utime.shape[0]
    lmax = hp.sphtfunc.Alm.getlmax(length_alm)
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(lmax)
    # uvw = compute_lnu(tracking=False)
    uvw = compute_pow_prasad()
    fig, axs = plot_prasad(uvw)
    """
    ufreq = np.fft.fft(utime, axis=1, norm="ortho")
    vfreq = np.fft.fft(vtime, axis=1, norm="ortho")
    wfreq = np.fft.fft(wtime, axis=1, norm="ortho")
    np.save(data_dir + "ufreq.npy", ufreq)
    np.save(data_dir + "vfreq.npy", vfreq)
    np.save(data_dir + "wfreq.npy", wfreq)
    np.save(data_dir + "freq.npy", freq_arr)


    _max = abs(ufreq).max()

    plt.figure()
    im = plt.imshow(ufreq.real, aspect=ufreq.shape[1]/ufreq.shape[0],
                    vmax=_max/100, vmin=-_max/100, cmap="seismic")
    plt.colorbar(im)
    plt.show()
    """

    """ # {{{ analyzing blocks
    block_size = 50
    analyze_blocks(utime, vtime, wtime, ellArr, emmArr, block_size, lmax)
    """ # }}} analyzing blocks

    ##  ust, vst, wst = get_st_plot(utime, vtime, wtime, ellArr, emmArr, 0, 50)
    ##  plot_all(ust, vst, wst, 0, 50)
#    smax = 140
#    ust, vst, wst = get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, smax)
#    ust, vst, wst = get_st_plot(utime, vtime, wtime, ellArr, emmArr, smax)
