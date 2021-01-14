import numpy as np
import matplotlib.pyplot as plt


# {{{ analyze_blocks_plot(u, comp, freqarr, num_blocks, lmax):
def analyze_blocks_plot(u, comp, freqarr, num_blocks, lmax):
    if comp==0:
        title_str = " $\\sqrt{\sum_{s, t} |B^r_{st}(\sigma)|^2}$ s $\in$ "
    elif comp==1:
        title_str = " $\sqrt{\sum_{s, t} s(s+1)|B^p_{st}(\sigma)|^2}$ s $\in$ "
    elif comp==2:
        title_str = " $\sqrt{\sum_{s, t} s(s+1)|B^t_{st}(\sigma)|^2}$ s $\in$ "

    block_size = lmax // num_blocks
    lmin, lmax = 0, 0
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))

    mask_pos = freqarr >= 0

    for i in range(num_blocks):
        print(f"Block number = {i+1} of {num_blocks}")
        lmin = lmax
        lmax = lmin + block_size
        xst = np.arange(lmax+1)
        axs.flatten()[i].semilogy(freqarr[mask_pos], abs(u[:, i])[mask_pos], 'black')
        axs.flatten()[i].set_title(title_str + f"({lmin}, {lmax})")
        axs.flatten()[i].set_xlabel(" $\sigma$ in $\mu$Hz")
        axs.flatten()[i].set_ylabel("Magnetic field in gauss")
        # axs.flatten()[i].legend()
    plt.tight_layout()
    fig.show()
    return None
# }}} analyze_blocks_plot(u, comp, freqarr, num_blocks, lmax)


# {{{ def sum_st(alm, ell, comp):
def sum_st(alm, ell, comp):
    if comp==0:
        return np.sqrt((abs(alm)**2)).sum()
    else:
        return np.sqrt((abs(alm)**2) * ell * (ell + 1)).sum()
# }}} def sum_st(alm, ell, comp):


# {{{ def analyze_blocks(u, comp, num_blocks, lmax):
def analyze_blocks(u, comp, num_blocks, lmax):
    block_size = lmax // num_blocks
    lmin, lmax = 0, 0
    u_block = np.zeros(num_blocks, dtype=np.float)

    for i in range(num_blocks):
        # print(f"Block number = {i+1} of {num_blocks}")
        lmin = lmax
        lmax = lmin + block_size
        mask_block = (ellArr <= lmax)*(ellArr > lmin)
        ell, alm = ellArr[mask_block], u[mask_block]
        u_block[i] = sum_st(alm, ell, comp)
    return u_block
# }}} analyze_blocks(u, comp, num_blocks, lmax)


# {{{ def get_alm(comp):
def get_alm(comp):
    if comp == 0:
        prefix = "BrlmA"
    elif comp == 1:
        prefix = "BplmA"
    elif comp == 2:
        prefix = "BtlmA"
    alm = np.load(f"{data_dir}/{prefix}.{suffix}.npy")
    return alm
# }}} get_alm(comp)


if __name__ == "__main__":

    max_dates = {"jan": 31, "feb": 28, "mar": 31, "apr": 30, "may": 31,
                "jun": 30, "jul": 31, "aug": 30}  #, "sep": 31}
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
            "jun": 6, "jul": 7, "aug": 8}  #, "sep": 9}
    data_dir = "/scratch/g.samarth/HMIDATA/magnetogram"
    arrlm = np.load(f"{data_dir}/arrlm.npz")
    ellArr, emmArr = arrlm['ellArr'], arrlm['emmArr']
    lmax = ellArr.max()
    tot_blocks = 8

    Br_time = np.zeros((400, tot_blocks), dtype=np.float)
    Bp_time = np.zeros((400, tot_blocks), dtype=np.float)
    Bt_time = np.zeros((400, tot_blocks),  dtype=np.float)
    time_count = 0

    for month in months:
        print(f"month = {month}")
        for i in range(1, max_dates[month]+1):
            try:
                suffix = f"2019{months[month]:02d}{i:02d}"
                Br, Bp, Bt = get_alm(0), get_alm(1), get_alm(2)
                Br_time[time_count, :] = analyze_blocks(Br, 0, tot_blocks, lmax)
                Bp_time[time_count, :] = analyze_blocks(Bp, 1, tot_blocks, lmax)
                Bt_time[time_count, :] = analyze_blocks(Bt, 2, tot_blocks, lmax)
                time_count += 1
            except FileNotFoundError:
                pass

    Br_time = Br_time[:time_count, :].copy()
    Bp_time = Bp_time[:time_count, :].copy()
    Bt_time = Bt_time[:time_count, :].copy()

    Br_freq = np.fft.fft(Br_time, axis=0)
    Bp_freq = np.fft.fft(Bp_time, axis=0)
    Bt_freq = np.fft.fft(Bt_time, axis=0)
    time_arr = np.arange(time_count)
    freq_arr = np.fft.fftfreq(len(time_arr), d=24*60*60)*1e6  # microHz

    analyze_blocks_plot(Br_freq, 0, freq_arr, tot_blocks, lmax)
    analyze_blocks_plot(Bp_freq, 1, freq_arr, tot_blocks, lmax)
    analyze_blocks_plot(Bt_freq, 2, freq_arr, tot_blocks, lmax)
