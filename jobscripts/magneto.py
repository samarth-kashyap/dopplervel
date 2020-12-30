import numpy as np

max_dates = {"jan": 31, "feb": 28, "mar": 31, "apr": 30, "may": 31,
             "jun": 30, "jul": 31, "aug": 30, "sep": 31}
months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
          "jun": 6, "jul": 7, "aug": 8, "sep": 9}

pythonpath = "/home/g.samarth/anaconda3/bin/python"
programpath = "/home/g.samarth/dopplervel2/get_vec_mag_spectra.py"

for month in months:
    for i in range(1, max_dates[month]+1):
        print(f"{pythonpath} {programpath} --gnup 2019{months[month]:02d}{i:02d}")
