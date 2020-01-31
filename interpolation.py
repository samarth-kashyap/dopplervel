from scipy.interpolate import griddata as GD
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy.io import fits
from math import sqrt, pi, e
import astropy.units as u
import numpy as np
import sunpy.map
import argparse
import time
import os

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

print("program working!!!")

if __name__=="__main__":
	print("here")
	hmiDataDir = scratchDir + "HMIDATA/v720s_dConS/2018/"
	hmiDataFileNames = hmiDataDir + "HMI_2018_filenames"

	with open(hmiDataFileNames, mode="r") as f:
		hmiFiles = f.read().splitlines()

	totalDays = len(hmiFiles)
	print(f"Total days = {totalDays}")

	"""	
	try:
		procid=int(os.environ['PBS_VNODENUM'])
	except KeyError: pass
	nproc = 6
	days = np.arange(procid, totalDays, nproc)
	print(f" procid = {procid}")
	"""

	gridNum = 400
	thetaUniGrid = np.linspace(1e-3, pi - 0.15, gridNum).reshape(gridNum, 1)
	phiUniGrid = np.linspace(-pi/2 + 1e-4, pi/2 - 1e-4, gridNum).reshape(1, gridNum)
	PHIUG, THETAUG = np.meshgrid(phiUniGrid, thetaUniGrid)

	for i in range(1):#totalDays):#days:
		print(i)
		hmi_map = sunpy.map.Map(hmiDataDir + hmiFiles[i])
		x, y = np.meshgrid(*[np.arange(v.value) for v in hmi_map.dimensions]) * u.pix
		hpc_coords = hmi_map.pixel_to_world(x, y)
		r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / hmi_map.rsun_obs

		rcrop = 0.999
		hmi_map.data[r>rcrop] = np.nan
		del x, y, r

		hpc_hgf = hpc_coords.transform_to(frames.HeliographicStonyhurst)
		theta = (hpc_hgf.lat.copy() + 90*u.deg).value * pi/180
		phi =  (hpc_hgf.lon.copy()).value * pi/180

		maskNaN = ~np.isnan(hmi_map.data)

		phi1D = phi[maskNaN]
		theta1D = theta[maskNaN]
		#mapData = hmi_map.data.copy()
		mapData = fits.open(hmiDataDir + "residual.fits", memmap=False)[0].data

		coords = np.zeros((phi1D.shape[0], 2))
		coords[:, 0] = theta1D
		coords[:, 1] = phi1D
		print("interpolating now")
		t2 = time.time()
		interpmap2 = GD(coords, mapData[maskNaN], (THETAUG, PHIUG), method='linear')

		# setting the invalid values to 0.0 (for spherical harmonic transfrom)
		masknan = ~np.isnan(interpmap2)
		interpmap2[~masknan] = 0.0
		cio.writefitsfile(interpmap2, hmiDataDir + 'int_' + hmiFiles[i] )
		t3 = time.time()
		print(f"time taken = {(t3-t2)/60} minutes")
		
		del phi1D, theta1D, mapData, coords, interpmap2, masknan, maskNaN, hmi_map, hpc_coords