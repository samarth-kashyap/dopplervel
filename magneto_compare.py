import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    ind_list = np.array([1, 51, 101, 151, 199], dtype=np.int)
    lmdir = "/scratch/g.samarth/HMIDATA/magnetogram"
    magdir = "/scratch/g.samarth/HMIDATA/data_analysis"
    flprefix = "alm.syn.inv.magneto"
    ellArr = np.load(f"{lmdir}/arrlm.npz")['ellArr']
    lmax = ellArr.max()

    regmin, regmax = 1e-6, 1e-2
    reglist = np.linspace(regmin, regmax, 200)

    # loading the actual spectrum
    ulm0 = np.load(f"{lmdir}/ulmA.magnetogram.npz")['ulm']
    vlm0 = np.load(f"{lmdir}/vlmA.magnetogram.npz")['vlm']
    wlm0 = np.load(f"{lmdir}/wlmA.magnetogram.npz")['wlm']
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

    """
    for i in ind_list:
        alm = np.load(f"{magdir}/{flprefix}.{i:02d}.npz"
        psu = computePS(alm['ulm'], lmax, ellArr, ellArr)
        psv = computePS(alm['vlm'], lmax, ellArr, ellArr)
        psw = computePS(alm['wlm'], lmax, ellArr, ellArr)
        diffu = psu - psu0
        diffv = psv - psv0
        diffw = psw - psw0
        axu.loglog(abs(diffu))
        axv.loglog(abs(diffv))
        axw.loglog(abs(diffw))
    axu.loglog(abs(psu0), 'r')
    axv.loglog(abs(psv0), 'r')
    axw.loglog(abs(psw0), 'r')
    figu.show()
    figv.show()
    figw.show()
    """
