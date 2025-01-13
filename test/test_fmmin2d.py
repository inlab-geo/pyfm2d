# this runs the original fmm simply turned into a subroutine that can be called
# from python like the fmm executable it reads the files from disk

from pyfm2dss import fastmarching as fmm

fmm.fmmin2d()
