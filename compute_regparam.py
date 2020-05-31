import argparse
import numpy as np
import matplotlib.pyplot as plt
from inversion import inv_reg1supp as reginv
from inversion import get_only_t

parser1 = argparse.ArgumentParser()
parser1.add_argument('--t', help="Argument for GNU Parallel",
                    type=int)
args = parser1.parse_args()

lmax = 1535
mat_dir = f"/scratch/g.samarth/matrixA/lmax{lmax}/"
t = args.t


def find_knee(reglist, leastsq):
    dy = leastsq[1:] - leastsq[:-1]
    dx = reglist[1:] - reglist[:-1]
    ls_d1 = dy/dx + 1
    prod = ls_d1[1:] * ls_d1[:-1]
    try:
        knee_ind = np.where(prod < 0)[0][0]
        knee = reglist[knee_ind]
    except IndexError:
        knee_ind = None
        knee = None
    return knee, knee_ind


def get_lcurve(t, regmax, Npoints, uA):
    A = np.load(f"{mat_dir}A{t:04d}.npz")['A']
    uot = A.dot(uA)

    regparam_list = np.linspace(1e-8, regmax, Npoints)
    leastsq = np.zeros(Npoints)
    for i in range(Npoints):
        if i % 5 == 0:
            print(f"{i} of {Npoints}")
        Ainv = reginv(A, regparam_list[i])
        uAt = Ainv.dot(uot)
        leastsq[i] = np.sqrt((abs(uAt - uA)**2).sum()/len(uA))

    fig = plt.figure(figsize=(7, 7))
    plt.loglog(regparam_list, leastsq)
    knee, knee_ind = find_knee(regparam_list, leastsq)
    if knee_ind:
        plt.loglog(knee, leastsq[knee_ind], 's', color="blue")
    plt.xlabel("Regularization parameter $\mu$")
    plt.ylabel("Residual")
    return regparam_list, leastsq, fig


def generate_synthetic(excitationType, maxIndex):
    ulm = np.zeros(maxIndex, dtype=complex)
    vlm = np.zeros(maxIndex, dtype=complex)
    wlm = np.zeros(maxIndex, dtype=complex)

    a, b = 5.0, 0.32
    countm, countl = 0, 0
    if excitationType == 'sectoral':
        for i in range(maxIndex):
            if (countl - countm) < 10:
                dml = countl - countm
                ulm[i] = a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
                vlm[i] = 10*a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
                wlm[i] = 10*a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1

    elif excitationType == 'tesseral':
        for i in range(maxIndex):
            if (countm > 7) and (countl - countm) < 7:
                dml = abs(countl/2 - countm)
                ulm[i] = a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
                vlm[i] = 10*a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
                wlm[i] = 10*a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1

    elif excitationType == 'zonal':
        for i in range(maxIndex):
            if countm < 10:
                dml = countm
                ulm[i] = a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
                vlm[i] = 10*a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
                wlm[i] = 10*a*countl/(a + np.exp(dml*b)) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1

    elif excitationType == 'solarlike':
        for i in range(maxIndex):
            logl = np.log(countl+1)
            ulm[i] = np.exp((60*logl - logl**2)/1000) *\
                ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            vlm[i] = np.exp((1.5*(60*logl - logl**2) + 1200)/1000) *\
                ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            wlm[i] = np.exp(1 + 0.04*logl) *\
                ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1

    elif excitationType == 'sparse':
        for i in range(maxIndex):
            a = np.random.rand()
            b = np.random.rand()
            c = np.random.rand()
            logl = np.log(countl+1)
            if a > 0.5:
                ulm[i] = np.exp((60*logl - logl**2)/1000) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            else:
                ulm[i] = 0.0
            if b > 0.5:
                vlm[i] = np.exp((1.5*(60*logl - logl**2) + 1200)/1000) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            else:
                vlm[i] = 0.0
            if c > 0.5:
                wlm[i] = np.exp(1 + 0.04*logl) *\
                    ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            else:
                wlm[i] = 0.0
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1

    elif excitationType == 'hathaway':
        ufit = np.load("u_poly.npz")['u']
        vfit = np.load("v_poly.npz")['v']
        wfit = np.load("w_poly.npz")['w']
        ellu = np.load("u_poly.npz")['ellu']
        ellv = np.load("v_poly.npz")['ellv']
        ellw = np.load("w_poly.npz")['ellw']
        for i in range(maxIndex):
            if countl < ellu[0].max():
                upoly = np.poly1d(ufit[0])
            else:
                upoly = np.poly1d(ufit[1])

            if countl < ellv[0].max():
                vpoly = np.poly1d(vfit[0])
            else:
                vpoly = np.poly1d(vfit[1])

            if countl < ellw[0].max():
                wpoly = np.poly1d(wfit[0])
            else:
                wpoly = np.poly1d(wfit[1])

            ulm[i] = upoly(countl) / np.sqrt(2*countl + 1) *\
                2 * (np.random.rand() + 1j*np.random.rand())
            vlm[i] = vpoly(countl) / np.sqrt(2*countl + 1) *\
                2 * (np.random.rand() + 1j*np.random.rand())
            wlm[i] = wpoly(countl) / np.sqrt(2*countl + 1) *\
                2 * (np.random.rand() + 1j*np.random.rand())
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1

    else:
        for i in range(maxIndex):
            ulm[i] = countl**0.75 *\
                ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            vlm[i] = 10*countl**0.75 *\
                ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            wlm[i] = 10*countl**0.75 *\
                ((np.random.rand() - 0.5) + 1j*(np.random.rand() - 0.5))
            if countm == countl:
                countm = 0
                countl += 1
            else:
                countm += 1
    return ulm, vlm, wlm


def get_lmarr(lmax):
    maxIndex = int((lmax+1)*(lmax+2)/2)
    ellArr = np.zeros(maxIndex, dtype=np.int32)
    emmArr = np.zeros(maxIndex, dtype=np.int32)
    countl, countm = 0, 0
    for i in range(maxIndex):
        ellArr[i] = countl
        emmArr[i] = countm
        if countm == countl:
            countm = 0
            countl += 1
        else:
            countm += 1
    return ellArr, emmArr


if __name__ == "__main__":

    maxIndex = int((lmax+1)*(lmax+2)/2)
    ellArr, emmArr = get_lmarr(lmax)
#    excitationType = 'sectoral'
#    excitationType = 'tesseral'
#    excitationType = 'zonal'
#    excitationType = 'random'
    excitationType = 'hathaway'
#    excitationType = 'solarlike'
#    excitationType = 'sparse'
    ulmAth, vlmAth, wlmAth = generate_synthetic(excitationType, maxIndex)
    u_actual_t = get_only_t(lmax, t, ulmAth, ellArr, emmArr)
    v_actual_t = get_only_t(lmax, t, vlmAth, ellArr, emmArr)
    w_actual_t = get_only_t(lmax, t, wlmAth, ellArr, emmArr)
    uA = np.concatenate((u_actual_t, v_actual_t, 1j*w_actual_t), axis=0)
    reglist, leastsq, fig = get_lcurve(t, 1e-7, 40, uA)
    plt.show()

    """
    if args.synth:
        # synthetic data has both true and observed alms
        arrFile = np.load("arrlm.npz")  # contains arrays of ell and emm
        almFile = np.load("alm.npz")    # contains arrays of alms - observed
        almAFile = np.load("almA.npz")  # contains arrays of alms - theoretical
        # loading ell and emm arrays
        ellArr = arrFile['ellArr']
        emmArr = arrFile['emmArr']
        # loading alms - observed
        ulmo = almFile['ulm']
        vlmo = almFile['vlm']
        wlmo = almFile['wlm']
        # loading alms - theoretical
        ulmAth = almAFile['ulm']
        vlmAth = almAFile['vlm']
        wlmAth = almAFile['wlm']
    """
