
# this uses the set,get and read functions

import numpy
import faulthandler

faulthandler.enable()

import pyfm2dss


fmm = pyfm2dss.FastMarchingMethod()

#  8     8                        c: Grid dicing in latitude and longitude
#  1                              c: Apply source grid refinement? (0=no,1=yes)
#  5     10                       c: Dicing level and extent of refined grid
#  6371.0                         c: Earth radius in km
#  1                              c: Use first-order(0) or mixed-order(1) scheme
#  0.5                            c: Narrow band size (0-1) as fraction of nnx*nnz


fmm.read_solver_options("fm2dss.in")
gdx,gdz,asgr,sgdl,sgs,earth,fom,snb,fsrt,cfd,wttf,wrgf=fmm.get_solver_options()
fmm.set_solver_options(gdx,gdz,asgr,sgdl,sgs,earth,fom,snb,fsrt,cfd,wttf,wrgf)


fmm.read_velocity_model("gridc.vtx")
nvx, nvz, goxd, gozd, dvxd, dvzd, velv=fmm.get_velocity_model()
fmm.set_velocity_model(nvx, nvz, goxd, gozd, dvxd, dvzd, velv)

fmm.read_sources("sources.dat")
scx, scz = fmm.get_sources()
fmm.set_sources(scx, scz)

fmm.read_receivers("receivers.dat")
rcx, rcz = fmm.get_receivers()
fmm.set_receivers(rcx, rcz)

fmm.read_source_receiver_associations("otimes.dat")
srs=fmm.get_source_receiver_associations()
fmm.set_source_receiver_associations(srs)

fmm.allocate_result_arrays()
fmm.run()





fmm.deallocate_result_arrays()