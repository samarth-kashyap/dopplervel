import numpy as np
import matplotlib.pyplot as plt

# {{{ def computePS(alm, lmax, ellArr, emmArr):
def computePS(alm, lmax, ellArr, emmArr):
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
        ps[i] += i * (abs(alm[isel])**2).sum()
    return np.sqrt(ps)
# }}} computePS(alm, lmax, ellArr, emmArr)


if __name__ == "__main__":
    datadir = "/scratch/g.samarth/HMIDATA/synth"
    LMAX = 130
    almA = np.load(f"{datadir}/almA.npz")
    ulmA, vlmA = almA['ulm'], almA['vlm']
    arrlm = np.load(f"{datadir}/arrlm.npz")
    ellArr, emmArr = arrlm['ellArr'], arrlm['emmArr']
    psuA = computePS(ulmA, LMAX, ellArr, emmArr)
    psvA = computePS(vlmA, LMAX, ellArr, emmArr)

    psu1 = np.load(f"{datadir}/psu1.npy")
    psu2 = np.load(f"{datadir}/psu2.npy")
    psu3 = np.load(f"{datadir}/psu3.npy")

    plt.figure()
    plt.plot((psu2[1:LMAX] - psu1[1:LMAX])*100/psu1[1:LMAX], 'k',
             label="$u_{\ell m}$ inverted from $(u^0_{st}, 2v^0_{st}, w^0_{st})$")
    plt.plot((psu3[1:LMAX] - psu1[1:LMAX])*100/psu1[1:LMAX], 'b',
             label="$u_{\ell m}$ inverted from $(u^0_{st}, 3v^0_{st}, w^0_{st})$")
    plt.xlabel("Spherical harmonic degree $\ell$", fontsize=15)
    plt.ylabel("Error \\%", fontsize=15)
    plt.title("Percentage error in inverted $u_{\ell m}$ compared \n to the inversion of " +
              "$u_{\ell m}$ from $(u^0_{st}, v^0_{st}, w^0_{st})$")
    plt.legend()
