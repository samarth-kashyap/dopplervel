import numpy as np
import matplotlib.pyplot as plt
plt.ion()

lmax = 250
daymax = 360
data_dir = "/scratch/g.samarth/HMIDATA/LCT"
arrlm = np.load(f"{data_dir}/arrlm.npz")
ellArr = arrlm['ellArr']; emmArr = arrlm['emmArr']
mask_lmax = ellArr <= lmax

timearr = np.linspace(0, daymax-1, daymax)*24*3600
freqarr = np.fft.fftfreq(len(timearr), d=timearr[1]-timearr[0])
freqarr *= 1e6 # converting to microHz
fpos_mask = (freqarr >= 0)*(freqarr <= 1)
sig_arr = freqarr[fpos_mask]
sig_len = len(sig_arr)


# {{{ def load_spectra():
def load_spectra():
    vst_time = np.zeros((mask_lmax.sum(), daymax), dtype=complex)
    wst_time = np.zeros((mask_lmax.sum(), daymax), dtype=complex)
    for i in range(daymax):
        temp = np.load(f"{data_dir}/almo_2011_{i:03d}.npz")
        vst_time[:, i] = temp['vlm'][mask_lmax]
        wst_time[:, i] = temp['wlm'][mask_lmax]
    return vst_time, wst_time
# }}} load_spectra()


def computePS_hathaway(vel_time):
    totpower = np.zeros(lmax)
    for ell in range(lmax):
        mask_ell = ellArr[mask_lmax]==ell
        totpower[ell] = 2* ell * (ell+1) * (abs(vel_time)[mask_ell]**2).sum()
    return np.sqrt(totpower/1e4)


# {{{ def computePS(vel_stf):
def computePS(vel_stf):
    totpower = np.zeros(lmax)
    for ell in range(lmax):
        if ell%2==1:
            mask_ell = ellArr[mask_lmax]==ell
            totpower[ell] = 2*ell*(ell+1)*(abs(vel_stf[mask_ell, :][:, fpos_mask])**2).sum()
    return np.sqrt(totpower)
# }}} computePS(vel_stf)


# {{{ def computePS_vs_sigma(vel_stf):
def computePS_vs_sigma(vel_stf):
    totpower = np.zeros(sig_len)
    for ell in range(lmax):
        if ell%2==1:
            mask_ell = ellArr[mask_lmax]==ell
            totpower += 2*ell*(ell+1)*(abs(vel_stf[mask_ell, :][:, fpos_mask])**2).sum(axis=0)
    return np.sqrt(totpower)
# }}} computePS_vs_sigma(vel_stf)


if __name__=="__main__":
    vst_time, wst_time = load_spectra()
    vst_freq = np.fft.fft(vst_time, axis=1)
    wst_freq = np.fft.fft(wst_time, axis=1)
    v_sig = computePS_vs_sigma(vst_freq)
    w_sig = computePS_vs_sigma(wst_freq)
    v_ell = computePS(vst_freq)
    w_ell = computePS(wst_freq)

    v_hath = computePS_hathaway(vst_time[:, 0])
    w_hath = computePS_hathaway(wst_time[:, 0])

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs[0][0].plot(sig_arr, v_sig, 'r')
    axs[0][0].plot(sig_arr, w_sig, 'b')
    axs[0][0].set_xlabel('frequency')
    axs[0][0].set_xlim([0, 2])

    axs[0][1].plot(v_ell, '.r')
    axs[0][1].plot(w_ell, '.b')
    axs[0][1].set_xlabel('ell')

    axs[1][0].loglog(v_hath, 'r')
    axs[1][0].loglog(w_hath, 'b')
    axs[1][0].set_xlabel('ell')
    plt.show(fig)

