import numpy as np
import argparse

# {{{ reading arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="year", type=int)
parser.add_argument("--chris", help="chris data", action="store_true")
args = parser.parse_args()
# }}} argparse

python_path = "/home/g.samarth/anaconda3/bin/python"
prog_path = "/home/g.samarth/dopplervel2/lct_process.py"
write_dir = "/home/g.samarth/dopplervel2/jobscripts/"
fname = f"ipjobs_lct_{args.year}.sh"

with open(write_dir + fname, "w") as ofile:
    for dn in range(365):
        valued_flags = f"--year {args.year} " + f"--daynum {dn}"
        if args.chris:
            valued_flags += " --chris"
        ofile.write(f"{python_path} {prog_path} {valued_flags}\n")
ofile.close()
