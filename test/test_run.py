
# this uses the set,get and read functions

import faulthandler

faulthandler.enable()

import pyfm2dss

fmm = pyfm2dss.FastMarchingMethod()
fmm.read_sources("sources.dat")
scx, scz = fmm.get_sources()
fmm.set_sources(scx, scz)
fmm.run()
