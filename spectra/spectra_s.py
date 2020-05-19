import numpy as np
import argparse
import matplotlib.pyplot as plt

def spectral_bands(alm, ellArr, bands):
    _lmax = ellArr.max()
    _last = alm.shape[0]
    num_bands = int(_lmax/bands)
    assert num_bands>0

    alm_band = np.zeros((num_bands, alm.shape[1]), \
                        dtype=float)
    mask_ell = np.zeros_like(ellArr, dtype=bool)
    for i in range(num_bands):
        mask_ell = (~mask_ell)*(ellArr<=bands*(i+1))
        count = mask_ell.sum()
        alm_band[i, :] = abs(alm[mask_ell, :]).sum(axis=0)/count

    return alm_band

def plot_bands(ulm, vlm, wlm, bands, ellArr):
    num_bands = int(ellArr.max()/bands)
    print(f"num_bands = {num_bands}, bands={bands}")
    fac = 2

    fig = plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 15})

    plt.subplot(311)
    _max = abs(ulm).max()
    im = plt.imshow(ulm, cmap='seismic',\
                    extent=[-5.753, 5.753, num_bands*bands, 0],\
                    aspect=11.4/(num_bands*bands), vmax=_max/fac)
    plt.colorbar(im)
    plt.xlabel('$\omega$ in $\mu$Hz')
    plt.ylabel('ell')
    plt.title("radial - bands")

    plt.subplot(312)
    _max = abs(vlm).max()
    im = plt.imshow(vlm, cmap='seismic',\
                    extent=[-5.753, 5.753, num_bands*bands, 0],\
                    aspect=11.4/(num_bands*bands), vmax=_max/fac)
    plt.colorbar(im)
    plt.xlabel('$\omega$ in $\mu$Hz')
    plt.ylabel('ell')
    plt.title("poloidal - bands")

    plt.subplot(313)
    _max = abs(wlm).max()
    im = plt.imshow(wlm, cmap='seismic',\
                    extent=[-5.753, 5.753, num_bands*bands, 0],\
                    aspect=11.4/(num_bands*bands), vmax=_max/fac)
    plt.colorbar(im)
    plt.xlabel('$\omega$ in $\mu$Hz')
    plt.title("toroidal - bands")
    plt.tight_layout()
    return fig


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', help="spherical harmonic degree",
                        type=int)
    parser.add_argument('--freqindex', help="index of frequency",
                        type=int)
    args = parser.parse_args()

    ellArr = np.load("ellArr.npy")
    emmArr = np.load("emmArr.npy")
    ufreq = np.load("ufreq.npy")
    vfreq = np.load("vfreq.npy")
    wfreq = np.load("wfreq.npy")
    nfreq = ufreq.shape[1]

    assert (args.s <= ellArr.max()), f"s larger than {ellArr.max()}"
    assert (args.freqindex < ufreq.shape[1]), f"freq index larger"\
        + f" than {ufreq.shape[1]}"

    u1 = ufreq[ellArr==args.s, :]
    v1 = vfreq[ellArr==args.s, :]
    w1 = wfreq[ellArr==args.s, :]

    bands = 30
    uband = spectral_bands(ufreq, ellArr, bands)
    vband = spectral_bands(vfreq, ellArr, bands)
    wband = spectral_bands(wfreq, ellArr, bands)
    fig = plot_bands(uband, vband, wband, bands, ellArr)
    fig.show()

    u1pmt = np.zeros((int(2*args.s), int(nfreq/2)), dtype=complex)
    v1pmt = np.zeros((int(2*args.s), int(nfreq/2)), dtype=complex)
    w1pmt = np.zeros((int(2*args.s), int(nfreq/2)), dtype=complex)
    tmask = ellArr==args.s
    tlist = emmArr[tmask]
    
    u1pmt[args.s:, :] = u1[:args.s, :int(nfreq/2)]
    u1pmt[:args.s, :] = u1[:args.s, int(nfreq/2 -1)::-1][::-1, :]

    v1pmt[args.s:, :] = v1[:args.s, :int(nfreq/2)]
    v1pmt[:args.s, :] = v1[:args.s, int(nfreq/2 -1)::-1][::-1, :]

    w1pmt[args.s:, :] = w1[:args.s, :int(nfreq/2)]
    w1pmt[:args.s, :] = w1[:args.s, int(nfreq/2 -1)::-1][::-1, :]

    plt.figure(figsize=(15, 5))
    plt.plot(emmArr[ellArr==args.s], abs(u1[:, args.freqindex]), 'g',
             label='radial')
    plt.plot(emmArr[ellArr==args.s], abs(v1[:, args.freqindex]), 'r',
             label='poloidal')
    plt.plot(emmArr[ellArr==args.s], abs(w1[:, args.freqindex]), 'b',
             label='toroidal')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 15})
    plt.subplot(311)
    _max = abs(u1pmt.real).max()/2
    im = plt.imshow(u1pmt.real, cmap='seismic',\
                    extent=[0,5.753, u1pmt.shape[0]-args.s, -args.s],\
                    aspect=5.753/2/args.s,\
                    vmax=_max, vmin=-_max)
    plt.colorbar(im)
    plt.xlabel('$\omega$ in $\mu$Hz')
    plt.ylabel('t')
    plt.title("radial")

    plt.subplot(312)
    _max = abs(v1pmt.real).max()/2
    im = plt.imshow(v1pmt.real, cmap='seismic',\
                    extent=[0,5.753, u1pmt.shape[0]-args.s, -args.s],\
                    aspect=5.753/2/args.s,\
                    vmax=_max, vmin=-_max)
    plt.colorbar(im)
    plt.xlabel('$\omega$ in $\mu$Hz')
    plt.ylabel('t')
    plt.title("poloidal")

    plt.subplot(313)
    _max = abs(w1pmt.real).max()/2
    im = plt.imshow(w1pmt.real, cmap='seismic',\
                    extent=[0,5.753, u1pmt.shape[0]-args.s, -args.s],\
                    aspect=5.753/2/args.s,\
                    vmax=_max, vmin=-_max)
    plt.colorbar(im)
    plt.xlabel('$\omega$ in $\mu$Hz')
    plt.ylabel('t')
    plt.title("toroidal")
    plt.tight_layout()
    plt.show()
