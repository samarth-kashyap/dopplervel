import numpy as np
pow1 = np.load("/scratch/g.samarth/dopplervel/synth_review/power.rot.new1.npz")
pow2 = np.load("/scratch/g.samarth/dopplervel/synth_review/power.rot.new2.npz")
pow3 = np.load("/scratch/g.samarth/dopplervel/synth_review/power.rot.new3.npz")
psv1 = pow1['vpow']
psv2 = pow2['vpow']
psv3 = pow3['vpow']
psu1 = pow1['upow']
psu2 = pow2['upow']
psu3 = pow3['upow']

