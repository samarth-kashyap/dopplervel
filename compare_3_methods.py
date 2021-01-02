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
    Power spectrum = sum_{m} | alm |^2

    """
    ps = np.zeros(lmax+1)
    for ell in range(lmax+1):
        index = np.where((ellArr == ell))[0]
        ps[ell] = 2 * (abs(coefs[index])**2).sum()*2 * ell # / (2*ell+1)
    return np.sqrt(ps)
# }}} computePS(ellArr, emmArr, lmax, coefs)


# {{{ def load_hath():
def load_hath(axs):
    ulm_hath = np.loadtxt("/scratch/g.samarth/dopplervel/datafiles/green.csv",
                          delimiter=",")
    vlm_hath = np.loadtxt("/scratch/g.samarth/dopplervel/datafiles/red.csv",
                          delimiter=",")
    wlm_hath = np.loadtxt("/scratch/g.samarth/dopplervel/datafiles/blue.csv",
                          delimiter=",")

    axs[0].loglog(vlm_hath[:, 0], vlm_hath[:, 1], 'b', label='hathaway', alpha=0.7)
    axs[1].loglog(wlm_hath[:, 0], wlm_hath[:, 1], 'b', label='hathaway', alpha=0.7)
    axs[2].loglog(ulm_hath[:, 0], ulm_hath[:, 1], 'b', label='hathaway', alpha=0.7)
    return axs

def load_lct(axs):
    lct_alm = np.load("/scratch/g.samarth/HMIDATA/LCT/alm_test.npz")
    lct_arrlm = np.load("/scratch/g.samarth/HMIDATA/LCT/arrlm.npz")
    lct_vlm, lct_wlm = lct_alm['vlm'], lct_alm['wlm']
    lct_ell, lct_emm = lct_arrlm['ellArr'], lct_arrlm['emmArr']
    psv_lct = computePS(lct_ell, lct_emm, lct_ell.max(), lct_vlm)
    psw_lct = computePS(lct_ell, lct_emm, lct_ell.max(), lct_wlm)

    axs[0].loglog(psv_lct, 'k', label='LCT', alpha=0.9)
    axs[1].loglog(psw_lct, 'k', label='LCT', alpha=0.9)
    return axs

def load_inv(axs, compute_u_by_tot=False):
    alm = np.load(f"{magdir}/lmax1535/alm.data.inv.final313.npz")
    ulm, vlm, wlm = alm['ulm'], alm['vlm'], alm['wlm']
    # only horizontal components need the scaling factor
    vlm *= np.sqrt(2)
    wlm *= np.sqrt(2)
    del alm
    ellArr, emmArr = hp.sphtfunc.Alm.getlm(hp.sphtfunc.Alm.getlmax(len(vlm)))
    lmax = 1535
    psv = computePS(ellArr, emmArr, lmax, vlm)
    psw = computePS(ellArr, emmArr, lmax, wlm)
    psu = computePS(ellArr, emmArr, lmax, ulm)
    axs[0].loglog(psv[:lmax_plot], color='red', label='inversion', alpha=0.7)
    axs[1].loglog(psw[:lmax_plot], color='red', label='inversion', alpha=0.7)
    axs[2].loglog(psu[:lmax_plot], color='red', label='inversion', alpha=0.7)
    if compute_u_by_tot:
        u_by_tot = psu / np.sqrt(psu**2 + psv**2 + psw**2)
        return axs, u_by_tot
    else:
        return axs

if __name__ == "__main__":
    magdir = "/scratch/g.samarth/HMIDATA/data_analysis"
    fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=3, sharex=True)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.08)
    fig.text(0.02, 0.50, 'Velocity \n  (ms${}^{-1}$)',
             va='center', rotation='horizontal', fontsize=14)
    fig.text(0.54, 0.01, " Spherical harmonic degree $s$ ",
             ha='center', fontsize=14)
    axs = load_hath(axs)
    axs = load_lct(axs)
    axs, u_by_tot = load_inv(axs, compute_u_by_tot=True)
    axs.flatten()[0].set_title("Poloidal Flow", fontsize=14)
    axs.flatten()[1].set_title("Toroidal Flow", fontsize=14)
    axs.flatten()[2].set_title("Radial Flow", fontsize=14)
    for i in range(3):
        axs.flatten()[i].tick_params(axis='both', which='major', labelsize=14)
        axs.flatten()[i].tick_params(axis='both', which='minor', labelsize=14)
    fig.tight_layout(rect=[0.08, 0.1, 1.0, 0.9])
    fig.savefig("/scratch/g.samarth/plots/doppler2_compare.pdf")
    # fig.show()

    fig2, ax2 = plt.subplots(1, figsize=(5, 4))
    ax2.plot(u_by_tot[:1500]*100, 'k', linewidth=0.5)
    ax2.set_ylim([1, 11])
    ax2.set_ylabel("\% Radial power", fontsize=10)
    ax2.set_xlabel("Spherical Harmonic degree $s$", fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='minor', labelsize=10)
    fig2.tight_layout()
    fig2.savefig("/scratch/g.samarth/plots/rad_power.pdf")
