import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cartopy

# The idea is to create high level classes to facilitate in inversio. Possible clases are
#
# - Survey to hold observations and predictions
# - Model to hold the velocity model 
# - WaveTracker to call FMM
# - Plotter plotting functions...
 
class WaveTracker:
	def __init__(self):	
		self.fmm=FastMarchingMethod()

class Survey:
	def __init__(self):
		pass
		
class Model:
	def __init__(self):
		pass

class Plotter:
    def __init__(self):
        pass
