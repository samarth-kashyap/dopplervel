from pyshtools import legendre as pleg
from scipy.integrate import simps
from math import sqrt, pi, e
from astropy.io import fits
import numpy as np
import argparse
import time
import os

# starting argument parser. 
# This section enables running program from daapc and hpc
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', help="Run program on cluster", action="store_true")
args = parser.parse_args()

if args.hpc:
	homeDir = "/home/samarth/"
	scratchDir = "/scratch/samarth/"
else:
	homeDir = "/home/samarthgk/hpchome/"
	scratchDir = "/home/samarthgk/hpcscratch/"

import sys; sys.path.append( homeDir )
from heliosPy import datafuncs as cdata
from heliosPy import mathfuncs as cmath
from heliosPy import iofuncs as cio

def get_pleg_index(l, m):
    """
    Gets the index for accessing legendre polynomials (generated from pyshtools.legendre)
    
    Inputs:
    l, m
    
    Outputs:
    index
    """
    return int(l*(l+1)/2 + m)

def gen_leg(lmax, theta):
	"""
	Generates associated legendre polynomials and derivatives

	Inputs:
	lmax      - Maximum spherical harmonic degree
	theta     - 1D array containing theta

	Outputs:
	leg       - Associated legendre polynomial
	dt_leg    - Derivative (d_theta) of associated legendre polynomial
	"""
	cost = np.cos(theta)
	sint = np.sin(theta)
	maxIndex = int( (lmax+1)*(lmax+2)/2 )
	leg = np.zeros( ( maxIndex, theta.size ) )
	leg_d1 = np.zeros( ( maxIndex, theta.size ) )

	ellArr = np.zeros(maxIndex)
	emmArr = np.zeros(maxIndex)
	count = 0
	for ell in range(lmax+1):
		for emm in range(ell+1):
			ellArr[count] = ell
			emmArr[count] = emm
			count += 1
	norm = np.sqrt( 2*ellArr*(ellArr+1) - emmArr*(2*ellArr + 1)).reshape(maxIndex, 1)

	count = 0
	for z in cost:
		leg[:, count], leg_d1[:, count] = pleg.PlmBar_d1(lmax, z, 1, 1)
		count += 1
	dt_leg = leg_d1 * (-sint).reshape(1, sint.shape[0])
	return leg, dt_leg/norm

if __name__=="__main__":		
	hmiDataDir = scratchDir + "HMIDATA/v720s_dConS/2018/"
	hmiDataFileNames = hmiDataDir + "HMI_2018_filenames"

	B0 = -0.053888227
	sB0 = np.sin(B0)
	cB0 = np.cos(B0)

	with open(hmiDataFileNames, mode="r") as f:
		hmiFiles = f.read().splitlines()

	totalDays = len(hmiFiles);
	print(f"Total days = {totalDays}")

	gridNum = 400
	thUG = np.linspace(1e-3, pi - 0.15, gridNum).reshape(gridNum, 1)
	phUG = np.linspace(-pi/2 + 1e-4, pi/2 - 1e-4, gridNum).reshape(1, gridNum)
	th1D = thUG.reshape(gridNum); ph1D = phUG.reshape(gridNum)
	phiUG, thetaUG = np.meshgrid(phUG, thUG)

	lmax = 200

	leg, dt_leg = gen_leg(lmax, th1D)
	cost = np.cos(thUG)
	sint = np.sin(thUG)
	cosp = np.cos(phUG)
	sinp = np.sin(phUG)

	lr = sB0 * cost + cB0 * sint * cosp
	lt = sB0 * sint - cB0 * cost * cosp
	lp = cB0 * sinp
	"""
	try:
		procid=int(os.environ['PBS_VNODENUM'])
	except KeyError: pass
	nproc = 120
	days = np.arange(procid, totalDays, nproc)
	print(f" procid = {procid}")
	"""

	for i in range(1):#days:#range(totalDays):
		print(f"Day = {i}")
		interpData = fits.open(hmiDataDir + "int_" + hmiFiles[i])[0].data

		velSize = int( (lmax + 1)*(lmax + 2)/2 )
		u = np.zeros(velSize, dtype=complex)
		v = np.zeros(velSize, dtype=complex)
		w = np.zeros(velSize, dtype=complex)
		count = 0
		t1 = time.time()
		for ell in range(lmax + 1):
			print(f" -- l = {ell}")
			for m in range(ell+1):
				cosmp = np.cos( - m * phUG)
				sinmp = np.sin( - m * phUG)
				eimp = cosmp + 1j*sinmp
				u[count] = simps( simps( sint * lr * interpData * \
					leg[get_pleg_index(ell, abs(m)), :] * eimp, axis=0, x=th1D), x=ph1D)

				v[count] = simps( simps( sint * interpData * \
					lt * dt_leg[get_pleg_index(ell, abs(m)), :] * eimp + \
						1j*m*interpData * lp * leg[get_pleg_index(ell, abs(m)), :] * eimp , axis=0, x=th1D), x=ph1D)

				w[count] = simps( simps( sint * interpData * lp * \
					dt_leg[get_pleg_index(ell, abs(m)), :] * eimp \
						- 1j*m*interpData * lt * leg[get_pleg_index(ell, abs(m)), :] * eimp, axis=0, x=th1D), x=ph1D)
				del cosmp, sinmp, eimp
				count += 1
		cio.writefitsfile(u.real, homeDir + 'u1_'+str(i).zfill(4)+'.fits')		
		cio.writefitsfile(u.imag, homeDir + 'u2_'+str(i).zfill(4)+'.fits')		
		cio.writefitsfile(v.real, homeDir + 'v1_'+str(i).zfill(4)+'.fits')		
		cio.writefitsfile(v.imag, homeDir + 'v2_'+str(i).zfill(4)+'.fits')		
		cio.writefitsfile(w.real, homeDir + 'w1_'+str(i).zfill(4)+'.fits')		
		cio.writefitsfile(w.imag, homeDir + 'w2_'+str(i).zfill(4)+'.fits')		
		del u, v, w
		t2 = time.time()
		print(f"{i}, Total time taken = {(t2 - t1)/60} minutes")