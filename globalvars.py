import numpy as np


class dopplervars():
    """Class to handle all the global variables and 
    parameters

    """
    def __init__(self):
        self.scratch = "/scratch/g.samarth"

    def get_dir(self, dirname):
        if dirname == "hmidata":
            return f"{self.scratch}/HMIDATA/v720s_dConS/2018/"
        if dirname == "plot":
            return f"{self.scratch}/plots/dopplervel/"
