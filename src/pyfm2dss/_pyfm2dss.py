import os
import glob
import numpy
import ctypes


class FastMarchingMethod:

    def __init__(self):
        self.libfm2dss = ctypes.cdll.LoadLibrary(
            glob.glob(os.path.dirname(__file__) + "/libfm2dss*.so")[0]
        )

    def fmmin2d(self):
        self.libfm2dss.fmmin2d()

    def run(self):
        self.libfm2dss.run()

    def read_sources(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_sources(fn, ctypes.c_int(len(fn.value)))

    def set_sources(self, scx_, scz_):
        nsrc_ = ctypes.c_int(len(scx_))
        #print(scx_,scz_)
        self.libfm2dss.set_sources(
            scx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nsrc_),
        )

    def get_sources(self):
        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc = nsrc_.value
        scx_ = numpy.empty((nsrc), dtype=ctypes.c_float)
        scz_ = numpy.empty((nsrc), dtype=ctypes.c_float)
        self.libfm2dss.get_sources(
            scx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nsrc_),
        )
        scx = numpy.array(scx_)
        scz = numpy.array(scz_)
        return scx, scz
