import os
import glob
import numpy
import ctypes

class FastMarchingMethod():
    
    def __init__(self):
        self.libfm2d = ctypes.cdll.LoadLibrary(glob.glob(os.path.dirname(__file__) + "/libfm2d*.so")[0])
    
    def run(self):
        self.libfm2d.fmmin2d()
    
