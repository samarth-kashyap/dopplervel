import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from math import sqrt, pi
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

lmax_plot = 1500

# {{{ def computePS(ellArr, emmArr, lmax, coefs):
def computePS(ellArr, emmArr, lmax, coefs):
    """Computes the power spectrum

    Parameters:
    -----------
    ellArr - np.ndarray(ndim=1, dtype=int)
        array containing the ell values
    emmArr - np.ndarray(ndim=1, dtype=int)
        array containing the emm values
    lmax - int
        maximum value of ell
    coefs - np.ndarray(ndim=1, dtype=complex)
        the alm coefficients

    Returns:
    --------
    ps - np.ndarray(ndim=1, dtype=float)
        power spectrum of the given alm

    Notes:
    ------
    Power spectrum = sum_{m=-l}^{l} l(l+1) | alm |^2

    """
    ps = np.zeros(lmax+1)
    for ell in range(lmax+1):
        mask_ell = ellArr == ell
        ps[ell] = 2 * (abs(coefs[mask_ell])**2).sum() * ell * (ell + 1)
    return np.sqrt(ps)
# }}} computePS(ellArr, emmArr, lmax, coefs)


def load_inv(axs):
    alm = np.load(f"{magdir}/lmax1535/alm.data.inv.final313.npz")
    wlm = alm['wlm']
    # only horizontal components need the scaling factor
    # needed for spin-1 harmonics convention in healPy
    wlm *= np.sqrt(2)
    del alm
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(hp.sphtfunc.Alm.getlmax(len(wlm)))
    lmax = 1535
    psw = computePS(ellArr, emmArr, lmax, wlm)
    ell = np.arange(lmax+1)
    ell[0] = 1
    axs.loglog(psw[:lmax_plot], color='black', label='inversion', rasterized=True)
    return axs


if __name__ == "__main__":
    magdir = "/scratch/g.samarth/HMIDATA/data_analysis"
    fig, axs = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.08)
    fig.text(0.015, 0.47, 'Convective \n Power \n (ms${}^{-1}$)',
             va='center', rotation='horizontal', fontsize=13)
    fig.text(0.54, 0.01, " Spherical harmonic degree $\ell$ ",
             ha='center', fontsize=14)
    axs = load_inv(axs)
    axs.set_title("$\sqrt{\sum\limits_{m=-\ell}^{\ell} \ell(\ell+1)|w_{\ell m}|^2}$", fontsize=13)
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.tick_params(axis='both', which='minor', labelsize=14)
    fig.tight_layout(rect=[0.15, 0.1, 1.0, 0.9])
    # fig.savefig("/scratch/g.samarth/plots/doppler2_compare.pdf")
