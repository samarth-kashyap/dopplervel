"""Visualize time series analysis of velocity field spherical harmonics.

This module provides plotting utilities for analyzing and displaying the
time-frequency characteristics of velocity field data. It creates diagnostic
plots showing power spectra as functions of frequency, spherical harmonic
degree, and azimuthal order.

The module supports both Doppler velocity and LCT data visualization with
customizable display options for different spectral components (radial,
poloidal, toroidal).
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from progress.bar import Bar
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

LMAX = 375
# LMAX = 1500

def get_title(comp, var_not_summed='sigma'):
    """Generate formatted LaTeX title string for plots.
    
    Creates appropriate mathematical notation for power spectrum plots
    based on the velocity component and variable being displayed.
    
    Parameters
    ----------
    comp : int
        Component index: 0=radial, 1=poloidal, 2=toroidal
    var_not_summed : str, optional
        Variable not being summed over: 'sigma', 't', or 's-|t|'
        Default is 'sigma'
    
    Returns
    -------
    title_str : str
        LaTeX formatted title string
    """
    if comp==0:
        if var_not_summed == 'sigma':
            title_str = " $\\sqrt{\sum\limits_{\\ell, m} |u_{\ell m}(\sigma)|^2}$ "
        elif var_not_summed == 't':
            title_str = " $\\sqrt{\sum\limits_{\ell, \sigma} |u_{\ell m}(\sigma)|^2}$ "
        elif var_not_summed == 's-|t|':
            title_str = " $\\sqrt{\sum\limits_{\ell, m, \sigma; \ell-|m|=\mathsf{const}}  |u_{\ell m}(\sigma)|^2}$ s $\in$ "
    elif comp==1:
        if var_not_summed == 'sigma':
            title_str = " $\\sqrt{\sum\limits_{\ell, m} \ell(\ell+1)|v_{\ell m}(\sigma)|^2}$ "
        elif var_not_summed == 't':
            title_str = " $\\sqrt{\sum\limits_{\ell, \sigma} \ell(\ell+1)|v_{\ell m}(\sigma)|^2}$ "
        elif var_not_summed == 's-|t|':
            title_str = " $\\sqrt{\sum\limits_{\ell, m, \sigma; \ell-|m|=\mathsf{const}}  \ell(\ell+1)|v_{\ell m}(\sigma)|^2}$ "
    elif comp==2:
        if var_not_summed == 'sigma':
            title_str = " $\\sqrt{\sum\limits_{\ell, m} \ell(\ell+1)|w_{\ell m}(\sigma)|^2}$ "
        elif var_not_summed == 't':
            title_str = " $\\sqrt{\sum\limits_{\ell, \sigma} \ell(\ell+1)|w_{\ell m}(\sigma)|^2}$ "
        elif var_not_summed == 's-|t|':
            title_str = " $\\sqrt{\sum\limits_{\ell, m, \sigma; \ell-|m|=\mathsf{const}}  \ell(\ell+1)|w_{\ell m}(\sigma)|^2}$ "
    return title_str

def get_title_new(comp, var_not_summed='sigma'):
    """Generate alternative formatted LaTeX title string for plots.
    
    Similar to get_title but uses different notation (s,t instead of ell,m).
    
    Parameters
    ----------
    comp : int
        Component index: 0=radial, 1=poloidal, 2=toroidal
    var_not_summed : str, optional
        Variable not being summed over: 'sigma', 't', or 's-|t|'
        Default is 'sigma'
    
    Returns
    -------
    title_str : str
        LaTeX formatted title string with (s,t) notation
    """
    if comp==0:
        if var_not_summed == 'sigma':
            title_str = " $\\left( {\sum\limits_{s, t} |u_{st}(\sigma)|^2 \\right)^{1/2}$ "
        elif var_not_summed == 't':
            title_str = " $\\left( \sum\limits_{s, \sigma} |u_{st}(\sigma)|^2 \\right)^{1/2}$ "
        elif var_not_summed == 's-|t|':
            title_str = " $\\left( \sum\limits_{s, t, \sigma; s-|t|=\mathsf{const}}  |u_{st}(\sigma)|^2 \\right)^{1/2}$ s $\in$ "
    elif comp==1:
        if var_not_summed == 'sigma':
            title_str = " $\\left( \sum\limits_{s, t} s(s+1)|v_{st}(\sigma)|^2 \\right)^{1/2}$ "
        elif var_not_summed == 't':
            title_str = " $\\left( \sum\limits_{s, \sigma} s(s+1)|v_{st}(\sigma)|^2 \\right)^{1/2}$ "
        elif var_not_summed == 's-|t|':
            title_str = " $\\left( \sum\limits_{s, t, \sigma; s-|t|=\mathsf{const}}  s(s+1)|v_{st}(\sigma)|^2 \\right)^{1/2}$ "
    elif comp==2:
        if var_not_summed == 'sigma':
            title_str = " $\\left( \sum\limits_{s, t} s(s+1)|w_{st}(\sigma)|^2 \\right)^{1/2}$ "
        elif var_not_summed == 't':
            title_str = " $\\left( \sum\limits_{s, \sigma} s(s+1)|w_{st}(\sigma)|^2 \\right)^{1/2}$ "
        elif var_not_summed == 's-|t|':
            title_str = " $\\left( \sum\limits_{s, t, \sigma; s-|t|=\mathsf{const}}  s(s+1)|w_{st}(\sigma)|^2 \\right)^{1/2}$ "
    return title_str


# {{{ analyze_blocks_plot(u, comp, num_blocks, var_not_summed='sigma'):
def analyze_blocks_plot(u, comp, num_blocks, var_not_summed='sigma', whichdata='lct',
                        figaxs=None):
    """Create multi-panel plots analyzing power in spherical harmonic degree blocks.
    
    Divides the spherical harmonic degree range into blocks and creates
    separate panels showing power distribution for each block. Supports
    overlaying multiple datasets (e.g., LCT vs Doppler).
    
    Parameters
    ----------
    u : np.ndarray
        Power array with shape (num_blocks, num_frequencies or num_orders)
    comp : int
        Component index: 0=radial, 1=poloidal, 2=toroidal
    num_blocks : int
        Number of spherical harmonic degree blocks to analyze
    var_not_summed : str, optional
        Variable displayed on x-axis: 'sigma', 't', or 's-|t|'
    whichdata : str, optional
        Data type identifier for styling: 'lct' or 'doppler'
    figaxs : tuple, optional
        Existing (fig, axs) to plot on. If None, creates new figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes objects
    """
    title_str = get_title(comp, var_not_summed)
    block_size = LMAX // num_blocks
    lmin, lmax = 0, 0
    ellmax_global = int(ellArr.max())
    tarr_global = np.arange(-ellmax_global, ellmax_global+1)
    smtp_global = np.arange(ellmax_global+1)
    if figaxs == None:
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05)
        fig.suptitle(title_str, fontsize=26, y=1.0, x=0.52)
        fig.text(0.02, 0.50, 'Convective power in m/s', va='center', rotation='vertical',
                 fontsize=26)
        if var_not_summed == 'sigma':
            fig.text(0.52, 0.01, " $\sigma$ in $\mu$Hz", ha='center', fontsize=26)
        elif var_not_summed == 't':
            fig.text(0.52, 0.01, " Azimuthal degree $m$ ", ha='center', fontsize=26)
        elif var_not_summed == 's-|t|':
            fig.text(0.52, 0.01, "$\ell - |m|$", ha='center', fontsize=26)
    else:
        fig, axs = figaxs

    pltsty = 'dashed' if whichdata == 'lct' else 'solid'
    pltclr = 'red' if whichdata == 'lct' else 'black'
    sigma_pos = freq_arr[freq_arr >= 0]

    bar = Bar(f'[{comp}-{whichdata}] - Plotting block number: ', max=num_blocks)
    for i in range(num_blocks):
        bar.next()
        lmin = lmax
        lmax = lmin + block_size
        xst = np.arange(lmax+1)

        if var_not_summed == 'sigma':
            axs.flatten()[i].plot(sigma_pos, abs(u[i, :]),
                                      color=pltclr,
                                      linestyle=pltsty)
            # umax, umin = abs(u[i, :]).max()*1.05, abs(u[i, :]).min()/10.

        elif var_not_summed == 't':
            axs.flatten()[i].plot(tarr_global, abs(u[i, :]),
                                      color=pltclr,
                                      linestyle=pltsty)
            if lmin == 0:
                axs.flatten()[i].set_xlim([-lmax-2, lmax+2])
            else:
                axs.flatten()[i].set_xlim([-lmin-2, lmin+2])
            masknan = ~np.isnan(u[i, :])
            # umax, umin = abs(u[i, masknan]).max()*1.05, abs(u[i, masknan]).min()/10.

        elif var_not_summed == 's-|t|':
            axs.flatten()[i].plot(smtp_global, abs(u[i, :]),
                                  color=pltclr,
                                  linestyle=pltsty)
            if lmin == 0:
                axs.flatten()[i].set_xlim([0, lmax+2])
            else:
                axs.flatten()[i].set_xlim([0, lmin+2])
            masknan = ~np.isnan(u[i, :])
            # umax, umin = abs(u[i, masknan]).max()*1.05, abs(u[i, masknan]).min()/10.

        if lmin > 0:
            axs.flatten()[i].set_title("$\ell \\in $ " + f"({lmin}, {lmax}];  " +
                                    "    $|m| \\le$ " f"{lmin}", fontsize=22)
        else:
            axs.flatten()[i].set_title("$\ell \\in $ " + f"({lmin}, {lmax}];  " +
                                    "    $|m| \\le$ " f"{lmax}", fontsize=22)
        # axs.flatten()[i].set_ylim([umin, umax])
        axs.flatten()[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs.flatten()[i].tick_params(axis='both', which='major', labelsize=22)
        axs.flatten()[i].tick_params(axis='both', which='minor', labelsize=22)
        # if figaxs == None:
            # ymin, ymax = axs.flatten()[i].get_ylim()
            # if ymax <= umax:
                # ymax = umax
            # if ymin >= umin:
                # ymin = umin
            # axs.flatten()[i].set_ylim([ymin, ymax])
        # axs.flatten()[i].legend()
    fig.tight_layout(rect=[0.05, 0.1, 1.0, 0.95])
    bar.finish()
    return fig, axs
# }}} analyze_blocks_plot(u, comp, num_blocks, lmax, var_not_summed='sigma')


# {{{ def sum_power(alm, comp, ell, emm, var_not_summed):
def sum_power(alm, comp, ell, emm, var_not_summed):
    mask_pos_freq = freq_arr >= 0
    mask_neg_freq = freq_arr <= 0
    sigma_pos = freq_arr[mask_pos_freq]
    sigma_neg = freq_arr[mask_neg_freq]
    lmin, lmax = int(ell.min()), int(ellArr.max())
    alm_sq_scaled = abs(alm)**2 if comp == 0 else ell*(ell+1)*abs(alm)**2
    if var_not_summed == 'sigma':
        totpower = np.sqrt(alm_sq_scaled[mask_pos_freq, :].sum(axis=1))

    elif var_not_summed == 't':
        tmax = int(emm.max())
        totpower = np.zeros(2*lmax+1, dtype=float)
        for tee in range(-tmax, tmax+1):
            masktee = emm == abs(tee)
            if tee >= 0:
                totpower[tee+lmax] = np.sqrt(alm_sq_scaled[mask_pos_freq, :]\
                                             [:, masktee].sum())
            else:
                totpower[tee+lmax] = np.sqrt(alm_sq_scaled[mask_neg_freq, :]\
                                             [:, masktee].sum())
        mask0 = totpower == 0
        totpower[mask0] = np.nan

    elif var_not_summed == 's-|t|':
        s_minus_tp = ell - emm
        if lmin > 0:
            mask_ext = s_minus_tp <= lmin
        else:
            mask_ext = np.ones_like(s_minus_tp, dtype=np.bool)

        totpower = np.zeros(lmax+1, dtype=float)
        smtpmin, smtpmax = s_minus_tp[mask_ext].min(), s_minus_tp[mask_ext].max()
        for smtp in range(smtpmin, smtpmax+1):
            mask_smtp = s_minus_tp == smtp
            totpower[smtp] = np.sqrt(alm_sq_scaled[:, mask_smtp].sum())
        mask0 = totpower == 0
        totpower[mask0] = np.nan

    elif var_not_summed == 's':
        totpower = np.zeros(lmax+1, dtype=float)
        for ell1 in range(lmax+1):
            maskell = ellArr==ell1
            totpower[ell1] = np.sqrt(alm_sq_scaled[:, maskell].sum())
        mask0 = totpower == 0
        totpower[mask0] = np.nan

    return totpower
# }}} sum_power(alm, comp, ell, emm, var_not_summed)


# {{{ def analyze_blocks(u, comp, num_blocks, var_not_summed='sigma'):
def analyze_blocks(u, comp, num_blocks, var_not_summed='sigma',
                   whichdata='lct'):
    block_size = LMAX // num_blocks
    lmin, lmax = 0, 0
    if var_not_summed == 'sigma':
        sigma_pos = freq_arr[freq_arr >= 0]
        u_block = np.zeros((num_blocks, len(sigma_pos)), dtype=np.float)
    elif var_not_summed == 't':
        ellmax_global = ellArr.max()
        u_block = np.zeros((num_blocks, 2*ellmax_global + 1), dtype=np.float)
    elif var_not_summed == 's-|t|':
        ellmax_global = ellArr.max()
        u_block = np.zeros((num_blocks, ellmax_global + 1), dtype=np.float)

    bar = Bar(f'[{comp}-{whichdata}] - Computing block number: ', max=num_blocks)
    for i in range(num_blocks):
        bar.next()
        lmin = lmax
        lmax = lmin + block_size
        if lmin > 0:
            mask_ell  = (ellArr <= lmax)*(ellArr > lmin)
            mask_emm = emmArr <= lmin
        else:
            mask_ell  = (ellArr <= lmax)*(ellArr >= lmin)
            mask_emm = mask_ell
        mask_block = mask_ell * mask_emm
        ell, emm, alm = ellArr[mask_block], emmArr[mask_block], u[:, mask_block]
        u_block[i] = sum_power(alm, comp, ell, emm, var_not_summed)
    bar.finish()
    return u_block
# }}} analyze_blocks(u, comp, num_blocks, var_not_summed='sigma')


# {{{ def get_alm(daynum):
def get_alm(daynum, whichdata='lct'):
    if whichdata == 'lct':
        alm = np.load(f"{data_dir}/almo_2011_{daynum:03d}.npz")
        return alm['vlm'][ellArr_lct <= LMAX], alm['wlm'][ellArr_lct <= LMAX]
    elif whichdata == 'inv':
        alm = np.load(f"{data_dir_inv}/alm.data.inv.final_test{daynum:03d}.npz")
        return alm['vlm'][ellArr_inv <= LMAX], alm['wlm'][ellArr_inv <= LMAX]
# }}} get_alm(daynum)


# {{{ def load_lct_data():
def load_lct_data(inv_exists):
    arrlen = len(ellArr)

    vst_time = np.zeros((daymax, arrlen), dtype=np.complex)
    wst_time = np.zeros((daymax, arrlen),  dtype=np.complex)
    time_count = 0

    bar = Bar(f'2. Loading LCT: ', max=daymax)
    for daynum in range(daymax):
        bar.next()
        try:
            inv_exists.index(daynum)
            try:
                vst, wst = get_alm(daynum, whichdata='lct')
                vst_time[time_count, :] = vst  #analyze_blocks(vst, 1, tot_blocks, lmax)
                wst_time[time_count, :] = wst  #analyze_blocks(wst, 2, tot_blocks, lmax)
                time_count += 1
            except FileNotFoundError:
                pass
        except ValueError:
            pass
    bar.finish()
    print(f"-- Computing FFT of LCT data ..")

    vst_time = vst_time[:time_count, :] * 1.0
    wst_time = wst_time[:time_count, :] * 1.0
    vst_sigma = np.fft.fft(vst_time, axis=0)/time_count
    wst_sigma = np.fft.fft(wst_time, axis=0)/time_count
    return vst_sigma, wst_sigma, time_count
# }}} load_lct_data():


def load_inv_data():
    arrlen = len(ellArr)
    inv_exists = []

    vst_time = np.zeros((daymax, arrlen), dtype=np.complex)
    wst_time = np.zeros((daymax, arrlen),  dtype=np.complex)
    time_count = 0

    bar = Bar(f'1. Loading inverted data: ', max=daymax)
    for daynum in range(daymax):
        bar.next()
        try:
            vst, wst = get_alm(daynum, whichdata='inv')
            vst_time[time_count, :] = vst  #analyze_blocks(vst, 1, tot_blocks, lmax)
            wst_time[time_count, :] = wst  #analyze_blocks(wst, 2, tot_blocks, lmax)
            inv_exists.append(daynum)
            time_count += 1
        except FileNotFoundError:
            pass
    bar.finish()
    print(f"-- Computing FFT of inverted data ..")

    vst_time = vst_time[:time_count, :] * np.sqrt(2)
    wst_time = wst_time[:time_count, :] * np.sqrt(2)
    vst_sigma = np.fft.fft(vst_time, axis=0)/time_count
    wst_sigma = np.fft.fft(wst_time, axis=0)/time_count
    return vst_sigma, wst_sigma, time_count, inv_exists


# {{{ def load_hath():
def load_hath(axs):
    ulm_hath = np.loadtxt("/scratch/g.samarth/dopplervel/datafiles/green.csv",
                          delimiter=",")
    vlm_hath = np.loadtxt("/scratch/g.samarth/dopplervel/datafiles/red.csv",
                          delimiter=",")
    wlm_hath = np.loadtxt("/scratch/g.samarth/dopplervel/datafiles/blue.csv",
                          delimiter=",")

    upow = ulm_hath[:, 1] / np.sqrt(ulm_hath[:, 0])
    vpow = vlm_hath[:, 1] * np.sqrt(vlm_hath[:, 0]+1)
    wpow = wlm_hath[:, 1] * np.sqrt(wlm_hath[:, 0]+1)

    axs.flatten()[0].loglog(vlm_hath[:, 0], vpow, '-.b', label='hathaway')
    axs.flatten()[1].loglog(wlm_hath[:, 0], wpow, '-.b', label='hathaway')
    axs.flatten()[2].loglog(ulm_hath[:, 0], upow, '-.b', label='hathaway')
    return axs
# }}} load_hath(axs)


def plot_power_spectrum():
    psv_lct = sum_power(vst_sigma_lct, 1, ellArr, emmArr, var_not_summed='s')
    psw_lct = sum_power(wst_sigma_lct, 2, ellArr, emmArr, var_not_summed='s')
    psu_inv = sum_power(ust_sigma_inv, 0, ellArr, emmArr, var_not_summed='s')
    psv_inv = sum_power(vst_sigma_inv, 1, ellArr, emmArr, var_not_summed='s')
    psw_inv = sum_power(wst_sigma_inv, 2, ellArr, emmArr, var_not_summed='s')
    hor_pow_lct = np.sqrt(psv_lct**2 + psw_lct**2)
    hor_pow_inv = np.sqrt(psv_inv**2 + psw_inv**2)

    ells = np.arange(len(psu_inv))

    fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
    fig.subplots_adjust(bottom=0.1, left=0.05)
    # fig.suptitle(, fontsize=26, y=1.0, x=0.52)
    fig.text(0.02, 0.50, 'Convective power in m/s',
             va='center', rotation='vertical', fontsize=26)
    fig.text(0.52, 0.01, "Spherical harmonic degree $s$",
             ha='center', fontsize=26)
    axs = load_hath(axs)

    axs.flatten()[0].loglog(psv_lct, 'k')
    axs.flatten()[0].loglog(psv_inv, 'r')
    axs.flatten()[0].set_title("Poloidal flow")

    axs.flatten()[1].loglog(psw_lct, 'k')
    axs.flatten()[1].loglog(psw_inv, 'r')
    axs.flatten()[1].set_title("Toroidal flow")

    axs.flatten()[2].loglog(psu_inv, 'r')
    axs.flatten()[2].set_title("Radial flow")

    fig.tight_layout(rect=[0.05, 0.1, 1.0, 1.0])
    return fig


if __name__ == "__main__":
    data_dir = "/scratch/g.samarth/HMIDATA/LCT"
    data_dir_inv = "/scratch/g.samarth/HMIDATA/data_analysis/lmax1535"
    lmax_inv = np.load(f"{data_dir_inv}/alm.data.inv.final_test005.npz")['ellmax']
    print(f"lmax_inv = {lmax_inv}")
    tot_blocks = 8
    daymax = 360

    arrlm = np.load(f"{data_dir}/arrlm.npz")
    ellArr_lct, emmArr_lct = arrlm['ellArr'], arrlm['emmArr']
    ellArr_inv, emmArr_inv = hp.sphtfunc.Alm.getlm(lmax_inv)
    ellArr = ellArr_lct[ellArr_lct <= LMAX]
    emmArr = emmArr_lct[ellArr_lct <= LMAX]
    var_list = ['t', 'sigma', 's-|t|']
    var_not_summed = 't'  # allowed options 'sigma', 't' and 's-|t|'

    vst_sigma_inv, wst_sigma_inv, time_count_inv, inv_exists = load_inv_data()
    vst_sigma_lct, wst_sigma_lct, time_count_lct = load_lct_data(inv_exists)
    assert time_count_lct == time_count_inv, "time counts don't match"
    time_count = time_count_lct
    time_arr = np.arange(time_count)
    freq_arr = np.fft.fftfreq(len(time_arr), d=24*60*60)*1e6  # microHz

    print("============================================================")

    for var_not_summed in var_list:
        print(f" == Plotting for var_not_summed = {var_not_summed} == ")
        # computing and plotting inverted data
        v_blocks = analyze_blocks(vst_sigma_inv, 1, tot_blocks,
                                  var_not_summed=var_not_summed,
                                  whichdata='inv')
        w_blocks = analyze_blocks(wst_sigma_inv, 2, tot_blocks,
                                  var_not_summed=var_not_summed,
                                  whichdata='inv')
        fig1, axs1 = analyze_blocks_plot(v_blocks*np.sqrt(2), 1, tot_blocks,
                                         var_not_summed=var_not_summed,
                                         whichdata='inv')
        fig2, axs2 = analyze_blocks_plot(w_blocks*np.sqrt(2), 2, tot_blocks,
                                         var_not_summed=var_not_summed,
                                         whichdata='inv')

        # computing and plotting LCT data
        v_blocks = analyze_blocks(vst_sigma_lct, 1, tot_blocks,
                                  var_not_summed=var_not_summed,
                                  whichdata='lct')
        w_blocks = analyze_blocks(wst_sigma_lct, 2, tot_blocks,
                                  var_not_summed=var_not_summed,
                                  whichdata='lct')
        fig1, axs1 = analyze_blocks_plot(v_blocks, 1, tot_blocks,
                                         var_not_summed=var_not_summed,
                                         whichdata='lct', figaxs=(fig1, axs1))
        fig2, axs2 = analyze_blocks_plot(w_blocks, 2, tot_blocks,
                                         var_not_summed=var_not_summed,
                                         whichdata='lct', figaxs=(fig2, axs2))

        fig1.savefig(f"/scratch/g.samarth/plots/figv_{var_not_summed}.pdf")
        fig2.savefig(f"/scratch/g.samarth/plots/figw_{var_not_summed}.pdf")

        del fig1, fig2, axs1, axs2
