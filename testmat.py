"""Test script for comparing old and new leakage matrix implementations.

This utility script loads two versions of the leakage matrix and generates
diagnostic plots to visualize differences between them. It helps validate
changes to the matrix generation code.
"""

import numpy as np
import matplotlib.pyplot as plt 

def plot_real_imag(Ao, An, matsize):
    """Plot real and imaginary parts of two matrices for comparison.
    
    Parameters
    ----------
    Ao : np.ndarray
        Old/original matrix
    An : np.ndarray
        New matrix to compare against
    matsize : int
        Number of rows to plot
    
    Returns
    -------
    int
        Status code (0 for success)
    """
    for i in range(matsize):
        plt.figure()
        plt.subplot(221)
        plt.plot(Ao[i, :].real, 'r')
        plt.plot(An[i, :].real, 'b')
        plt.title('real part')
        
        plt.subplot(222)
        plt.plot(Ao[i, :].imag, 'r')
        plt.plot(An[i, :].imag, 'b')
        plt.title('imag part')

        plt.subplot(223)
        plt.plot( Ao[i, :].real - An[i, :].real, 'r')
        plt.title("diff: real part")
        
        plt.subplot(224)
        plt.plot( Ao[i, :].imag- An[i, :].imag, 'r')
        plt.title("diff: imag part")
        plt.show()
    return 0

def plot_abs(Ao, An, matsize):
    """Plot absolute values of two matrices for comparison.
    
    Parameters
    ----------
    Ao : np.ndarray
        Old/original matrix
    An : np.ndarray
        New matrix to compare against
    matsize : int
        Number of rows to plot
    
    Returns
    -------
    int
        Status code (0 for success)
    """
    for i in range(matsize):
        plt.figure()
        plt.subplot(211)
        plt.plot(abs(Ao[i, :]), 'r')
        plt.plot(abs(An[i, :]), 'b')
        plt.title('magnitude')
        
        plt.subplot(212)
        plt.plot( abs(Ao[i, :]) - abs(An[i, :]), 'r')
        plt.title('difference')
        plt.show()
    return 0



Ao = np.load("mat_old.npz")['A']
An = np.load("mat_new.npz")['A']
matsize = Ao.shape[0]

diff = abs(Ao) - abs(An)
_max = abs(diff).max()
plt.figure()
plt.subplot(221)
im = plt.imshow(abs(Ao), cmap='seismic')
plt.colorbar(im)
plt.subplot(222)
im = plt.imshow(abs(An), cmap='seismic')
plt.colorbar(im)
plt.subplot(223)
im = plt.imshow(abs(Ao) - abs(An), vmax=_max, vmin=-_max, cmap='seismic')
plt.colorbar(im)
plt.show()
plot_real_imag(Ao, An, matsize)