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
        # print(scx_,scz_)
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

    def read_receivers(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_receivers(fn, ctypes.c_int(len(fn.value)))

    def set_receivers(self, rcx_, rcz_):
        nrc_ = ctypes.c_int(len(rcx_))
        self.libfm2dss.set_receivers(
            rcx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rcz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nrc_),
        )

    def get_receivers(self):
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc = nrc_.value
        rcx_ = numpy.empty((nrc), dtype=ctypes.c_float)
        rcz_ = numpy.empty((nrc), dtype=ctypes.c_float)
        self.libfm2dss.get_sources(
            rcx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rcz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nrrc_),
        )
        rcx = numpy.array(rcx_)
        rcz = numpy.array(rcz_)
        return scx, scz

    def read_source_receiver_associations(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_source_receiver_associations(
            fn, ctypes.c_int(len(fn.value))
        )

    def set_source_receiver_associations(self,srs_):
        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc=nsrc_.value
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc=nrc_.value
        
        srs_ = numpy.asfortranarray(numpy.zeros([nsrc, nrc]), dtype=numpy.int32)
        srs_ = self.libfm2dss.set_source_receiver_associations(
            srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        srs = numpy.array(srs_)
        return srs

    def get_source_receiver_associations(self):
        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc=nsrc_.value
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc=nrc_.value
        
        srs_ = numpy.asfortranarray(numpy.zeros([nsrc, nrc]), dtype=numpy.int32)
        self.libfm2dss.get_source_receiver_associations(
            srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
