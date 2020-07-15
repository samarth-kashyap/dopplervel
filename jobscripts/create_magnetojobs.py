import numpy as np

python_path = "/home/g.samarth/anaconda3/envs/py3/bin/python"
prog_path = "/home/g.samarth/dopplervel2/inversion.py"
write_dir = "/home/g.samarth/dopplervel2/jobscripts/"
fname = "ipjobs_magneto.sh"
flags = "--synth --magneto --cchpc"
reg_min, reg_max = 1e-6, 1e-2
reg_arr_log = np.linspace(np.log(reg_min), np.log(reg_max), 200)
reg_arr = np.exp(reg_arr_log)

with open(write_dir + fname, "w") as ofile:
    for i, regvalue in enumerate(reg_arr):
        valued_flags = f"--gnup {i} " + f"--reg {regvalue:0.7e}"
        ofile.write(f"{python_path} {prog_path} {flags} {valued_flags}\n")
ofile.close()
