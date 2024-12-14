# this runs the original fmm simply turned into a subroutine that can be called 
# from python like the fmm executable it reads the files from disk

import pyfm2dss

fmm = pyfm2dss.FastMarchingMethod()
fmm.fmmin2d()
