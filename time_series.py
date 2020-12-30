# {{{ Library imports
import matplotlib.pyplot as plt            # Plotting
from heliosPy import iofuncs as cio
import healpy as hp
import numpy as np
import argparse
# }}} imports


# {{{ ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on daahpc",
                    action="store_true")
parser.add_argument('--cchpc', help="Run program on cchpc19",
                    action="store_true")
parser.add_argument('--datatype', help="\'LCT\' or \'doppler\'",
                    type=str)
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
    data_dir = "/scratch/g.samarth/HMIDATA/LCT/"

NTIME = 2 #300


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
            fname = f"almo_2011_{i:03d}.npz"
        try:
            alm = np.load(fname)
            alm_found = True
        except FileNotFoundError:
            print(f"{fname} not found")
            alm_found = False
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
            if datatype == 'doppler':
                utime[:, i] = alm['ulm']
            vtime[:, i] = alm['vlm']
            wtime[:, i] = alm['wlm']
            if i%10==0:
                print(f" i = {i} ")
    if datatype == 'doppler':
        return utime, vtime, wtime
    elif datatype == 'LCT':
        return vtime, wtime
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


if __name__=="__main__":
    max_time = NTIME*24*3600  # total time in seconds
    time_arr = np.linspace(0, max_time, NTIME)
    dtime = time_arr[1] - time_arr[0]
    freq_arr = np.fft.fftfreq(time_arr.shape[0], dtime)
    dfreq = freq_arr[1] - freq_arr[0]

    utime, vtime, wtime = load_time_series(NTIME, data_dir,
                                           datatype=args.datatype)
    length_alm = utime.shape[0]
    lmax = hp.sphtfunc.Alm.getlmax(length_alm)
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(lmax)
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

    block_size = 50
    analyze_blocks(utime, vtime, wtime, ellArr, emmArr, block_size, lmax)

    ##  ust, vst, wst = get_st_plot(utime, vtime, wtime, ellArr, emmArr, 0, 50)
    ##  plot_all(ust, vst, wst, 0, 50)
#    smax = 140
#    ust, vst, wst = get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, smax)
#    ust, vst, wst = get_st_plot(utime, vtime, wtime, ellArr, emmArr, smax)
