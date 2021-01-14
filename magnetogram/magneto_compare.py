import numpy as np
import matplotlib.pyplot as plt
import sys
import healpy as hp


# {{{ def computePS(alm, lmax, ellArr, emmArr):
def computePS(alm, lmax, ellArr):
    '''Computes the power spectrum given the spectral coefficients.

    Parameters:
    -----------
    alm - np.ndarray(ndim=1, dtype=complex)
        array of all spectral coefficients (upto lmax)
    lmax - int
        maximum spherical harmonic degree
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing ell
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing emm

    Returns:
    --------
    ps - np.ndarray(ndim=1, dtype=float)
        power spectrum

    Notes:
    ------
    The velocity power spectrum is given by ell * sum_{m} | alm |^2

    '''
    ps = np.zeros(lmax)
    for i in range(lmax):
        isel = ellArr == i
        ps[i] += (abs(alm[isel])**2).sum() * i  # / (2*i + 1)
    return np.sqrt(ps)
# }}} computePS(alm, lmax, ellArr, emmArr)


# {{{ def diffsq(alm, alm0):
def diffsq(alm, alm0):
    s1 = ((alm[0] - alm0[0])**2).sum()
    s2 = ((alm[1] - alm0[1])**2).sum()
    s3 = ((alm[2] - alm0[2])**2).sum()
    return s1, s2, s3
# }}} diffsq(alm, alm0)


# {{{ def plot_all(ust, vst, wst, lmin, lmax):
def plot_all(uth, uobs, lmin, lmax, st_or_t):
    ust, vst, wst = uobs
    ustA, vstA, wstA = uth
    xst = np.arange(lmax+1)

    if st_or_t:
        xlab = "$s - |t|$"
    else:
        xlab = "t"

    ylab = "Velocity in ms$^{-1}$" if synth else "Magnetic field in gauss"

    dirname = syndir if synth else lmdir
    if lct:
        dirname = lctdir

    plt.figure(figsize=(10, 10))
#    plt.rcParams.update({'font.size': 20})
    plt.title(" Total power summed over $smin$ =" +
              f" {lmin}," +"$smax$ = " + f"{lmax}")
    if not lct:
        plt.subplot(221)
        plt.semilogy(xst, ust, '--', label='radial - inverted')
        plt.semilogy(xst, ustA, 'black', label='radial - actual')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()

    plt.subplot(222)
    plt.semilogy(xst, vst, '--', label='poloidal - inverted')
    plt.semilogy(xst, vstA, 'black', label='poloidal - actual')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

    plt.subplot(223)
    plt.semilogy(xst, wst, '--', label='toroidal - inverted')
    plt.semilogy(xst, wstA, 'black', label='toroidal - actual')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

    if st_or_t:
        plot_fname = f"{dirname}/st_log_{lmin:04d}_{lmax:04d}.png"
    else:
        plot_fname = f"{dirname}/t_log_{lmin:04d}_{lmax:04d}.png"
    print(f"Saving plot to {plot_fname}")
    plt.savefig(plot_fname)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.title(" Total power summed over $smin$ =" +
              f" {lmin}," +"$smax$ = " + f"{lmax}")
    if not lct:
        plt.subplot(221)
        plt.plot(xst, ust, '--', label='radial - inverted')
        plt.plot(xst, ustA, 'black', label='radial - actual')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()

    plt.subplot(222)
    plt.plot(xst, vst, '--', label='poloidal - inverted')
    plt.plot(xst, vstA, 'black', label='poloidal - actual')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

    plt.subplot(223)
    plt.plot(xst, wst, '--', label='toroidal - inverted')
    plt.plot(xst, wstA, 'black', label='toroidal - actual')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()


    if st_or_t:
        plot_fname = f"{dirname}/st_{lmin:04d}_{lmax:04d}.png"
    else:
        plot_fname = f"{dirname}/t_{lmin:04d}_{lmax:04d}.png"
    print(f"Saving plot to {plot_fname}")
    plt.savefig(plot_fname)
    plt.close()
    return None
# }}} plot_all(ust, vst, wst, lmin, lmax)


# {{{ def get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, lmin, lmax):
def get_st_plot(u, v, w, ellArr, emmArr, lmin, lmax, st_or_t):
    '''Takes the spectral coefficients and returns an array for a given s

    Parameters:
    -----------
    u - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic
    v - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic
    w - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic
    ellArr - np.ndarray(ndim=1, dtype=int)
        list of ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        list of emm values
    lmax - int
        maximum value of ell upto which power is summed

    Returns:
    --------
    u_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic for s, t
    v_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of poloidal spherical harmonic for s, t
    w_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of toroidal spherical harmonic for s, t

    '''
    len_alm = u.shape[0]
    u_st = np.zeros(lmax+1)
    v_st = np.zeros(lmax+1)
    w_st = np.zeros(lmax+1)
    for st in range(lmax+1):
        if st_or_t:
            mask = (ellArr - emmArr == st) * (ellArr < lmax) * (ellArr >= lmin)
        else:
            mask = (emmArr == st) * (ellArr < lmax) * (ellArr >= lmin)
        u_st[st] += abs(u[mask]).sum()
        v_st[st] += abs(v[mask]).sum()
        w_st[st] += abs(w[mask]).sum()
    return u_st, v_st, w_st
# }}} get_st_plot(ufreq, vfreq, wfreq, ellArr, emmArr, lmin, lmax)


# {{{ def get_st_single(u, ellArr, emmArr, lmin, lmax):
def get_st_single(u, comp, ellArr, emmArr, lmin, lmax, st_or_t):
    '''Takes the spectral coefficients and returns an array for a given s

    Parameters:
    -----------
    u - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic
    ellArr - np.ndarray(ndim=1, dtype=int)
        list of ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        list of emm values
    lmax - int
        maximum value of ell upto which power is summed

    Returns:
    --------
    u_st - np.ndarray(ndim=2, dtype=complex)
        frequency series of radial spherical harmonic for s, t

    '''
    len_alm = u.shape[0]
    u_st = np.zeros(lmax+1)
    for st in range(lmax+1):
        if st_or_t:
            mask = (ellArr - emmArr == st) * (ellArr < lmax) * (ellArr >= lmin)
        else:
            mask = (emmArr == st) * (ellArr < lmax) * (ellArr >= lmin)
        if comp == 0:
            u_st[st] += (abs(u[mask])**2).sum()
        else:
            ell = ellArr[mask].copy()
            u_st[st] += (ell*(ell+1) * abs(u[mask])**2).sum()
#### u_st[st] += (abs(u[mask])**2).sum()
    return np.sqrt(u_st)
# }}} get_st_single(u, ellArr, emmArr, lmin, lmax)


# {{{ def analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax):
def analyze_blocks(uth, uobs, ellArr, emmArr, num_blocks, lmax):
    u, v, w = uobs
    uth, vth, wth = uth
    block_size = lmax // num_blocks
    lmin, lmax = 0, 0
    st_or_t = True
    for i in range(num_blocks-1):
        print(f"Block number = {i+1} of {num_blocks-1}")
        lmin = lmax
        lmax = lmin + block_size
        ust, vst, wst = get_st_plot(u, v, w,
                                    ellArr, emmArr, lmin, lmax, st_or_t)
        ustA, vstA, wstA = get_st_plot(uth, vth, wth,
                                       ellArr, emmArr, lmin, lmax, st_or_t)
        plot_all((ustA, vstA, wstA), (ust, vst, wst), lmin, lmax, st_or_t)
    return None
# }}} analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax)


# {{{ analyze_blocks_plot(ulist, ellArr, emmArr, num_blocks, lmax):
def analyze_blocks_plot(ulist, comp, ellArr, emmArr, num_blocks, lmax):
    u, u0 = ulist
    if comp==0:
        title_str = " $\sqrt{|B^r_{st}|^2}$ for s $\in$ "
    elif comp==1:
        title_str = " $\sqrt{s(s+1) |B^p_{st}|^2}$ for s $\in$ "
    elif comp==2:
        title_str = " $\sqrt{s(s+1) |B^t_{st}|^2}$ for s $\in$ "

    block_size = lmax // num_blocks
    lmin, lmax = 0, 0
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    
    plot_count = 0

    st_or_t = True
    for i in range(num_blocks):
        print(f"Block number = {i+1} of {num_blocks}")
        lmin = lmax
        lmax = lmin + block_size
        xst = np.arange(lmax+1)
        ust = get_st_single(u0, comp, ellArr, emmArr, lmin, lmax, st_or_t)
        # ust0 = get_st_single(u0, comp, ellArr, emmArr, lmin, lmax, st_or_t)
        axs.flatten()[plot_count].semilogy(xst, ust, 'black', label='ME\_720s\_fd')
        # axs.flatten()[plot_count].semilogy(xst, ust0,
                                           # '--', color='blue', label='inversion from LOS')
        axs.flatten()[plot_count].set_title(title_str + f"({lmin}, {lmax})")
        axs.flatten()[plot_count].set_xlabel(" $s - |t|$")
        axs.flatten()[plot_count].set_ylabel("Magnetic field in gauss")
        # axs.flatten()[plot_count].legend()
        plot_count += 1

    st_or_t = False
    lmin, lmax = 0, 0
    for i in range(num_blocks):
        print(f"Block number = {i+1} of {num_blocks}")
        lmin = lmax
        lmax = lmin + block_size
        xst = np.arange(lmax+1)
        ust = get_st_single(u0, comp, ellArr, emmArr, lmin, lmax, st_or_t)
        # ust0 = get_st_single(u0, comp, ellArr, emmArr, lmin, lmax, st_or_t)
        axs.flatten()[plot_count].semilogy(xst, ust, 'black', label='ME\_720s\_fd')
        # axs.flatten()[plot_count].semilogy(xst, ust0,
                                           # '--', color='blue', label='inversion from LOS')
        axs.flatten()[plot_count].set_title(title_str + f"({lmin}, {lmax})")
        axs.flatten()[plot_count].set_xlabel(" $t$")
        axs.flatten()[plot_count].set_ylabel("Magnetic field in gauss")
        # axs.flatten()[plot_count].legend()
        plot_count += 1
    plt.tight_layout()
    if rempel:
        fname = f"mag.{comp}.png"
    else:
        fname = f"vel.{comp}.png"
    fig.savefig(f"/scratch/g.samarth/plots/vecmagneto/{fname}")
    return fig
# }}} analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax)


# {{{ analyze_blocks_plot_vel(ulist, ellArr, emmArr, num_blocks, lmax):
def analyze_blocks_plot_vel(u, comp, ellArr, emmArr, num_blocks, lmax):
    if comp==0:
        title_str = " $\sqrt{|u_{st}|^2}$ for s $\in$ "
    elif comp==1:
        title_str = " $\sqrt{s(s+1) |v_{st}|^2}$ for s $\in$ "
    elif comp==2:
        title_str = " $\sqrt{s(s+1) |w_{st}|^2}$ for s $\in$ "

    block_size = lmax // num_blocks
    lmin, lmax = 0, 0
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    
    plot_count = 0

    st_or_t = True
    for i in range(num_blocks):
        print(f"Block number = {i+1} of {num_blocks}")
        lmin = lmax
        lmax = lmin + block_size
        xst = np.arange(lmax+1)
        ust = get_st_single(u, comp, ellArr, emmArr, lmin, lmax, st_or_t)
        axs.flatten()[plot_count].plot(xst, ust, 'black')
        axs.flatten()[plot_count].set_title(title_str + f"({lmin}, {lmax})")
        axs.flatten()[plot_count].set_xlabel(" $s - |t|$")
        axs.flatten()[plot_count].set_ylabel("Velocity in m/s")
        plot_count += 1

    st_or_t = False
    lmin, lmax = 0, 0
    for i in range(num_blocks):
        print(f"Block number = {i+1} of {num_blocks}")
        lmin = lmax
        lmax = lmin + block_size
        xst = np.arange(lmax+1)
        ust = get_st_single(u, comp, ellArr, emmArr, lmin, lmax, st_or_t)
        axs.flatten()[plot_count].plot(xst, ust, 'black')
        axs.flatten()[plot_count].set_title(title_str + f"({lmin}, {lmax})")
        axs.flatten()[plot_count].set_xlabel(" $t$")
        axs.flatten()[plot_count].set_ylabel("Velocity in m/s")
        plot_count += 1
    plt.tight_layout()
    if rempel:
        fname = f"mag.{comp}.png"
    elif inv:
        fname = f"vel.{comp}.png"
    elif lct:
        fname = f"lct.{comp}.png"
    fig.savefig(f"/scratch/g.samarth/plots/vecmagneto/{fname}")
    return fig
# }}} analyze_blocks(ufreq, vfreq, wfreq, ellArr, emmArr, block_size, lmax)


# {{{ def computePS(alm, lmax, ellArr, emmArr):
def computePS(alm, comp, lmax, ellArr, emmArr):
    '''Computes the power spectrum given the spectral coefficients.

    Parameters:
    -----------
    alm - np.ndarray(ndim=1, dtype=complex)
        array of all spectral coefficients (upto lmax)
    lmax - int
        maximum spherical harmonic degree
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing ell
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing emm

    Returns:
    --------
    ps - np.ndarray(ndim=1, dtype=float)
        power spectrum

    Notes:
    ------
    The velocity power spectrum is given by ell * \sum_{m} | alm |^2

    '''
    ps = np.zeros(lmax)
    for i in range(lmax):
        isel = ellArr == i
        if comp == 0:
            ps[i] += (abs(alm[isel])**2).sum()
        else:
            ps[i] += (i*(i+1))*(abs(alm[isel])**2).sum()
    return np.sqrt(ps)
# }}} computePS(alm, lmax, ellArr, emmArr)


# {{{ def plot_inv_actual(inv, act, ell, args):
def plot_inv_actual(inv, act, ell, args):
    if rempel:
        yaxis_label = "Magnetic field in G"
        title_suffix = "magnetic field"
    else:
        yaxis_label = "Velocity in m/s"
        title_suffix = "velocity"
    inv_total = np.sqrt(inv[0]**2 + inv[1]**2 + inv[2]**2)
    act_total = np.sqrt(act[0]**2 + act[1]**2 + act[2]**2)
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 15})
    plt.subplot(221)
    plt.xlabel("Spherical harmonic degree $s$")
    plt.ylabel(yaxis_label)
    ts2 = "$\sqrt{\sum_s |B^r_{st}|^2}$" if rempel else \
        "$\sqrt{\sum_s |u_{st}|^2}$"
    plt.title(f"Radial {title_suffix}: " + ts2)
    plt.loglog(inv[0], 'g-.', label='inverted from LOS')

    plt.subplot(222)
    ts2 = "$\sqrt{\sum_s s(s+1) |B^p_{st}|^2}$" if rempel else \
        "$\sqrt{\sum_s s(s+1) |v_{st}|^2}$"
    plt.title(f"Poloidal {title_suffix}: " + ts2)
    plt.xlabel("Spherical harmonic degree $s$")
    plt.ylabel(yaxis_label)
    plt.loglog(inv[1], 'r-.', label='inverted from LOS')

    plt.subplot(223)
    ts2 = "$\sqrt{\sum_s s(s+1) |B^t_{st}|^2}$" if rempel else \
        "$\sqrt{\sum_s s(s+1) |w_{st}|^2}$"
    plt.title(f"Toroidal {title_suffix}: " + ts2)
    plt.xlabel("Spherical harmonic degree $s$")
    plt.ylabel(yaxis_label)
    plt.loglog(inv[2], 'b-.', label='inverted from LOS')

    plt.subplot(224)
    plt.title(f"Total {title_suffix}")
    plt.xlabel("Spherical harmonic degree $s$")
    plt.ylabel(yaxis_label)
    plt.loglog(inv_total, color='black',
               linestyle='-.', label='inverted')
    plt.tight_layout()
    return fig
# }}} plot_inv_actual(inv, act, ell, args)


if __name__ == "__main__":
    ind_list = np.array([1, 51, 101, 151, 199], dtype=np.int)
    lmdir = "/scratch/g.samarth/HMIDATA/magnetogram"
    magdir = "/scratch/g.samarth/HMIDATA/data_analysis"
    syndir = "/scratch/g.samarth/HMIDATA/synth"
    lctdir = "/scratch/g.samarth/HMIDATA/LCT"
    flprefix = "alm.syn.inv.magneto"

    reg_test = False
    compute_ps = True
    synth = False
    lct = False
    rempel = False
    inv = True

    if rempel:
        alm = np.load(f"{magdir}/{flprefix}.196.npz")
        ellArr = np.load(f"{lmdir}/arrlm.npz")['ellArr']
        emmArr = np.load(f"{lmdir}/arrlm.npz")['emmArr']
        lmax = ellArr.max()
        ulm, vlm, wlm = alm['ulm'], alm['vlm'], alm['wlm']
        print(f" alm = {len(ulm)}, ellArr = {len(ellArr)}")
        ulm0 = np.load(f"{lmdir}/ulmA.magnetogram.npz")['ulm']
        vlm0 = np.load(f"{lmdir}/vlmA.magnetogram.npz")['vlm']
        wlm0 = np.load(f"{lmdir}/wlmA.magnetogram.npz")['wlm']
        # fig = analyze_blocks_plot((ulm, ulm0), 0, ellArr, emmArr, 8, lmax)
        # fig = analyze_blocks_plot((vlm, vlm0), 1, ellArr, emmArr, 8, lmax)
        # fig = analyze_blocks_plot((wlm, wlm0), 2, ellArr, emmArr, 8, lmax)
        psu = computePS(ulm0, 0, lmax, ellArr, emmArr)
        psv = computePS(vlm0, 1, lmax, ellArr, emmArr)
        psw = computePS(wlm0, 2, lmax, ellArr, emmArr)
        psvel = (psu, psv, psw)
        fig = plot_inv_actual(psvel, psvel, np.arange(lmax), 0)
        fig.show()
        sys.exit()
    elif inv:
        alm = np.load(f"{magdir}/lmax1535/alm.data.inv.final313.npz")
        ulm, vlm, wlm = alm['ulm'], alm['vlm'], alm['wlm']
        del alm
        ellArr, emmArr = hp.sphtfunc.Alm.getlm(hp.sphtfunc.Alm.getlmax(len(ulm)))
        lmax = 1535
        fig = analyze_blocks_plot_vel(ulm, 0, ellArr, emmArr, 8, lmax)
        fig.show()
        fig = analyze_blocks_plot_vel(vlm, 1, ellArr, emmArr, 8, lmax)
        fig.show()
        fig = analyze_blocks_plot_vel(wlm, 2, ellArr, emmArr, 8, lmax)
        fig.show()
        # del fig
        # psu = computePS(ulm, 0, lmax, ellArr, emmArr)
        # psv = computePS(vlm, 1, lmax, ellArr, emmArr)
        # psw = computePS(wlm, 2, lmax, ellArr, emmArr)
        # psvel = (psu, psv, psw)
        # fig = plot_inv_actual(psvel, psvel, np.arange(lmax), 0)
        # fig.show()
        sys.exit()
    elif lct:
        alm = np.load("/scratch/g.samarth/HMIDATA/LCT/almo.npz")
        arrlm = np.load("/scratch/g.samarth/HMIDATA/LCT/arrlm.npz")
        vlm, wlm = alm['vlm'], alm['wlm']
        ellArr, emmArr = arrlm['ellArr'], arrlm['emmArr']
        lmax = ellArr.max()
        fig = analyze_blocks_plot_vel(wlm, 2, ellArr, emmArr, 8, lmax)
        fig.show()
        fig = analyze_blocks_plot_vel(vlm, 1, ellArr, emmArr, 8, lmax)
        fig.show()
        # analyze_blocks((vlm, vlm, wlm), (vlm, vlm, wlm),
                       # ellArr, emmArr, 8, lmax)
        sys.exit()

    """
    ellArr = np.load(f"{lmdir}/arrlm.npz")['ellArr']
    emmArr = np.load(f"{lmdir}/arrlm.npz")['emmArr']
    lmax = ellArr.max()

    if synth:
        alm0 = np.load("/scratch/g.samarth/HMIDATA/synth/almA.npz")
        alm = np.load("/scratch/g.samarth/HMIDATA/data_analysis/alm.syn.inv..npz")
        arrlm = np.load("/scratch/g.samarth/HMIDATA/synth/arrlm.npz")
        uth = (alm0['ulm'], alm0['vlm'], alm0['wlm'])
        uobs = (alm['ulm'], alm['vlm'], alm['wlm'])
        ellArr, emmArr = arrlm['ellArr'], arrlm['emmArr']
        lmax = ellArr.max()
        analyze_blocks(uth, uobs, ellArr, emmArr, 10, lmax)
        exit()

    ulm0 = np.load(f"{lmdir}/ulmA.magnetogram.npz")['ulm']
    vlm0 = np.load(f"{lmdir}/vlmA.magnetogram.npz")['vlm']
    wlm0 = np.load(f"{lmdir}/wlmA.magnetogram.npz")['wlm']

    if compute_ps:
        alm = np.load(f"{magdir}/{flprefix}.196.npz")
        ulm, vlm, wlm = alm['ulm'], alm['vlm'], alm['wlm']
        psu = computePS(ulm, lmax, ellArr, emmArr)
        psv = computePS(vlm, lmax, ellArr, emmArr)
        psw = computePS(wlm, lmax, ellArr, emmArr)

        psuth = computePS(ulm0, lmax, ellArr, emmArr)
        psvth = computePS(vlm0, lmax, ellArr, emmArr)
        pswth = computePS(wlm0, lmax, ellArr, emmArr)

        fig = plot_inv_actual((psu, psv, psw), (psuth, psvth, pswth),
                              np.arange(lmax), 0)
        fig.show()
        exit()

    if reg_test:
        print("wrong")
        regmin, regmax = 1e-6, 1e-2
        reglist = np.linspace(regmin, regmax, 200)

        # loading the actual spectrum
        psu0 = computePS(ulm0, lmax, ellArr)
        psv0 = computePS(vlm0, lmax, ellArr)
        psw0 = computePS(wlm0, lmax, ellArr)
        ps0 = (psu0, psv0, psw0)
        diffu = np.zeros(200)
        diffv = np.zeros(200)
        diffw = np.zeros(200)

        for i in range(1, 200):
            if i % 25 == 1:
                print(i)
            alm = np.load(f"{magdir}/{flprefix}.{i:02d}.npz")
            ulm, vlm, wlm = alm['ulm'], alm['vlm'], alm['wlm']
            psu = computePS(ulm, lmax, ellArr)
            psv = computePS(vlm, lmax, ellArr)
            psw = computePS(wlm, lmax, ellArr)
            ps = (psu, psv, psw)
            diffu[i], diffv[i], diffw[i] = diffsq(ps, ps0)

        plt.figure()
        plt.plot(reglist, np.sqrt(diffu), 'r')
        plt.plot(reglist, np.sqrt(diffv), 'g')
        plt.plot(reglist, np.sqrt(diffw), 'b')
        plt.plot(reglist, np.sqrt(diffu+diffv+diffw), 'b')
        plt.show()

    else:
        print("here")
        alm = np.load(f"{magdir}/{flprefix}.196.npz")
        ulm, vlm, wlm = alm['ulm'], alm['vlm'], alm['wlm']
        uobs = (ulm, vlm, wlm)
        uth = (ulm0, vlm0, wlm0)
        analyze_blocks(uth, uobs, ellArr, emmArr, 100, lmax)

    """
