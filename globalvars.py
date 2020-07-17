"""Contains class to handle global variables

"""

class DopplerVars():
    """Class to handle all the global variables and 
    parameters

    """

    def __init__(self):
        self.scratch = "/scratch/g.samarth"
        self.home = "/home/g.samarth"
        return None

    def get_dir(self, dirname):
        if dirname == "hmidata":
            return f"{self.scratch}/HMIDATA/v720s_dConS/2018/"
        if dirname == "plot":
            return f"{self.scratch}/plots/dopplervel/"
        return None

# adding lines from iMac
# adding line from cchpc
