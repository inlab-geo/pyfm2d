
# this uses the set,get and read functions

import faulthandler

faulthandler.enable()

import pyfm2dss

fmm = pyfm2dss.FastMarchingMethod()
fmm.read_sources("sources.dat")
scx, scz = fmm.get_sources()
fmm.set_sources(scx, scz)

fmm.read_receivers("receivers.dat")
rcx, rcz = fmm.get_sources()
fmm.set_receivers(rcx, rcz)

fmm.read_source_receiver_associations("otimes.dat")
srs=fmm.get_source_receiver_associations()
fmm.set_source_receiver_associations(srs)

fmm.run()
