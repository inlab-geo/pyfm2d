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

    def read_solver_options(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_configuration(fn, ctypes.c_int(len(fn.value)))

    def set_solver_options(
        self, gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf
    ):
        gdx_ = ctypes.c_int(gdx)
        gdz_ = ctypes.c_int(gdz)
        asgr_ = ctypes.c_int(asgr)
        sgdl_ = ctypes.c_int(sgdl)
        sgs_ = ctypes.c_int(sgs)
        earth_ = ctypes.c_float(earth)
        fom_ = ctypes.c_int(fom)
        snb_ = ctypes.c_int(snb)

        fsrt_ = ctypes.c_int(fsrt)
        cfd_ = ctypes.c_int(cfd)
        wttf_ = ctypes.c_int(wttf)
        wrgf_ = ctypes.c_int(wrgf)

        self.libfm2dss.set_solver_options(
            ctypes.byref(gdx_),
            ctypes.byref(gdz_),
            ctypes.byref(asgr_),
            ctypes.byref(sgdl_),
            ctypes.byref(sgs_),
            ctypes.byref(earth_),
            ctypes.byref(fom_),
            ctypes.byref(snb_),
            ctypes.byref(fsrt_),
            ctypes.byref(cfd_),
            ctypes.byref(wttf_),
            ctypes.byref(wrgf_),
        )

    def get_solver_options(self):
        gdx_ = ctypes.c_int(-99)
        gdz_ = ctypes.c_int(-99)
        asgr_ = ctypes.c_int(-99)
        sgdl_ = ctypes.c_int(-99)
        sgs_ = ctypes.c_int(-99)
        earth_ = ctypes.c_float(-99.9)
        fom_ = ctypes.c_int(-99)
        snb_ = ctypes.c_int(-99)

        fsrt_ = ctypes.c_int(-99)
        cfd_ = ctypes.c_int(-99)
        wttf_ = ctypes.c_int(-99)
        wrgf_ = ctypes.c_int(-99)

        self.libfm2dss.get_solver_options(
            ctypes.byref(gdx_),
            ctypes.byref(gdz_),
            ctypes.byref(asgr_),
            ctypes.byref(sgdl_),
            ctypes.byref(sgs_),
            ctypes.byref(earth_),
            ctypes.byref(fom_),
            ctypes.byref(snb_),
            ctypes.byref(fsrt_),
            ctypes.byref(cfd_),
            ctypes.byref(wttf_),
            ctypes.byref(wrgf_),
        )

        gdx = gdx_.value
        gdz = gdz_.value
        asgr = asgr_.value
        sgdl = sgdl_.value
        sgs = sgs_.value
        earth = earth_.value
        fom = fom_.value
        snb = snb_.value

        fsrt = fsrt_.value
        cfd = cfd_.value
        wttf = wttf_.value
        wrgf = wrgf_.value

        return gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf

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
        self.libfm2dss.get_receivers(
            rcx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rcz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nrc_),
        )
        rcx = numpy.array(rcx_)
        rcz = numpy.array(rcz_)
        return rcx, rcz

    def read_source_receiver_associations(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_source_receiver_associations(
            fn, ctypes.c_int(len(fn.value))
        )

    def set_source_receiver_associations(self, srs):
        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc = nsrc_.value
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc = nrc_.value

        srs_ = numpy.asfortranarray(srs, dtype=numpy.int32)
        srs_ = self.libfm2dss.set_source_receiver_associations(
            srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        srs = numpy.array(srs_)
        return srs

    def get_source_receiver_associations(self):

        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc = nsrc_.value
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc = nrc_.value

        srs_ = numpy.asfortranarray(numpy.zeros([nsrc, nrc]), dtype=numpy.int32)
        self.libfm2dss.get_source_receiver_associations(
            srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        srs = srs_
        return srs_

    def read_velocity_model(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_velocity_model(fn, ctypes.c_int(len(fn.value)))

    def set_velocity_model(self, nvx, nvz, goxd, gozd, dvxd, dvzd, velv):
        nvx_ = ctypes.c_int(nvx)
        nvz_ = ctypes.c_int(nvz)
        goxd_ = ctypes.c_float(goxd)
        gozd_ = ctypes.c_float(gozd)
        dvxd_ = ctypes.c_float(dvxd)
        dvzd_ = ctypes.c_float(dvzd)
        velv_ = numpy.asfortranarray(velv, dtype=numpy.float32)

        self.libfm2dss.set_velocity_model(
            ctypes.byref(nvx_),
            ctypes.byref(nvz_),
            ctypes.byref(goxd_),
            ctypes.byref(gozd_),
            ctypes.byref(dvxd_),
            ctypes.byref(dvzd_),
            velv_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def get_velocity_model(self):
        nvx_ = ctypes.c_int(-99)
        nvz_ = ctypes.c_int(-99)

        self.libfm2dss.get_number_of_velocity_model_vertices(
            ctypes.byref(nvx_), ctypes.byref(nvz_)
        )

        goxd_ = ctypes.c_float(-99.9)
        gozd_ = ctypes.c_float(-99.9)

        dvxd_ = ctypes.c_float(-99.9)
        dvzd_ = ctypes.c_float(-99.9)

        velv_ = numpy.asfortranarray(
            numpy.zeros([nvx_.value + 1, nvz_.value + 1]), dtype=numpy.float32
        )

        self.libfm2dss.get_velocity_model(
            ctypes.byref(nvx_),
            ctypes.byref(nvz_),
            ctypes.byref(goxd_),
            ctypes.byref(gozd_),
            ctypes.byref(dvxd_),
            ctypes.byref(dvzd_),
            velv_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        nvx = nvx_.value
        nvz = nvz_.value

        goxd = goxd_.value
        gozd = gozd_.value

        dvxd = dvxd_.value
        dvzd = dvzd_.value

        return nvx, nvz, goxd, gozd, dvxd, dvzd, velv_
