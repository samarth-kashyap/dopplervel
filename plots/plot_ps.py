import numpy as np 
import matplotlib.pyplot as plt 

def ps(ulm, vlm, wlm, lmaxCalc, ellArr):
    ulmGrid = np.zeros((lmaxCalc+1, lmaxCalc+1), dtype=complex)
    vlmGrid = np.zeros((lmaxCalc+1, lmaxCalc+1), dtype=complex)
    wlmGrid = np.zeros((lmaxCalc+1, lmaxCalc+1), dtype=complex)
    for ell in range(lmaxCalc):
        elmask = ellArr==ell
        ulmGrid[ell, :(ell+1)] = ulm[elmask]
        vlmGrid[ell, :(ell+1)] = vlm[elmask]
        wlmGrid[ell, :(ell+1)] = wlm[elmask]
    return ulmGrid, vlmGrid, wlmGrid

arrFile = np.load("arrlm.npz")
almFile = np.load("almA.npz")
#almAFile = np.load("almA.npz")
almAFile = np.load("almInv.npz")
ellArr, emmArr = arrFile['ellArr'], arrFile['emmArr']
ulmo, vlmo, wlmo = almFile['ulm'], almFile['vlm'], almFile['wlm']
ulmA, vlmA, wlmA = almAFile['ulm'], almAFile['vlm'], almAFile['wlm']

lmaxCalc = 40
uoG, voG, woG = ps(ulmo, vlmo, wlmo, lmaxCalc, ellArr)
uAG, vAG, wAG = ps(ulmA, vlmA, wlmA, lmaxCalc, ellArr)

plt.figure()
plt.subplot(321)
im = plt.imshow(abs(uoG), cmap='seismic')
plt.colorbar(im)
plt.title('ulm - actual')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.subplot(322)
im = plt.imshow(abs(uAG), cmap='seismic')
plt.colorbar(im)
plt.title('ulm - inverted')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.subplot(323)
im = plt.imshow(abs(voG), cmap='seismic')
plt.colorbar(im)
plt.title('vlm - actual')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.subplot(324)
im = plt.imshow(abs(vAG), cmap='seismic')
plt.colorbar(im)
plt.title('vlm - inverted')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.subplot(325)
im = plt.imshow(abs(woG), cmap='seismic')
plt.colorbar(im)
plt.title('wlm - actual')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.subplot(326)
im = plt.imshow(abs(wAG), cmap='seismic')
plt.colorbar(im)
plt.title('wlm - inverted')
plt.xlabel("$m$")
plt.ylabel("$l$")


plt.tight_layout()
plt.show()

totpowo = np.sqrt( abs(uoG)**2 + abs(voG)**2 + abs(woG)**2 )
totpowA = np.sqrt( abs(uAG)**2 + abs(vAG)**2 + abs(wAG)**2 )
_max = max(abs(totpowo).max(), abs(totpowo).min())
fac = 2.0

plt.figure()
plt.subplot(131)
im = plt.imshow(totpowo, cmap='seismic', vmax=_max)
plt.colorbar(im)
plt.title('total power - actual')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.subplot(132)
im = plt.imshow(fac*totpowA, cmap='seismic', vmax=_max)
plt.colorbar(im)
plt.title('total power - inverted')
plt.xlabel("$m$")
plt.ylabel("$l$")
diffpow = totpowo - fac*totpowA
plt.subplot(133)
im = plt.imshow(diffpow, cmap='seismic', vmax=_max)
plt.colorbar(im)
plt.title('total power - diff')
plt.xlabel("$m$")
plt.ylabel("$l$")

plt.tight_layout()
plt.show()

print(f"diffpow max = {abs(diffpow[:20, :20]).max()}")

