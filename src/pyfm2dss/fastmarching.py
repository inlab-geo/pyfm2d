import os
import glob
import numpy as np
import scipy
import ctypes


class FastMarchingMethod:

    def __init__(self):
        self.libfm2dss = ctypes.cdll.LoadLibrary(
            glob.glob(os.path.dirname(__file__) + "/libfm2dss*.so")[0]
        )

    def fmmin2d(self):
        self.libfm2dss.fmmin2d()

    def track(self):
        self.libfm2dss.track()

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
        snb_ = ctypes.c_float(snb)

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
        snb_ = ctypes.c_float(-99)

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
        scx_ = np.empty((nsrc), dtype=ctypes.c_float)
        scz_ = np.empty((nsrc), dtype=ctypes.c_float)
        self.libfm2dss.get_sources(
            scx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nsrc_),
        )
        scx = np.array(scx_)
        scz = np.array(scz_)
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
        rcx_ = np.empty((nrc), dtype=ctypes.c_float)
        rcz_ = np.empty((nrc), dtype=ctypes.c_float)
        self.libfm2dss.get_receivers(
            rcx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rcz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nrc_),
        )
        rcx = np.array(rcx_)
        rcz = np.array(rcz_)
        return rcx, rcz

    def read_source_receiver_associations(self, fn_):
        fn = ctypes.c_char_p(fn_.encode("UTF-8"))
        self.libfm2dss.read_source_receiver_associations(
            fn, ctypes.c_int(len(fn.value))
        )

    def set_source_receiver_associations(self, srs):
        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))

        srs_ = np.asfortranarray(srs, dtype=np.int32)
        srs_ = self.libfm2dss.set_source_receiver_associations(
            srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        return

    def get_source_receiver_associations(self):

        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc = nsrc_.value

        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc = nrc_.value

        srs_ = np.asfortranarray(np.zeros([nrc, nsrc]), dtype=np.int32)
        self.libfm2dss.get_source_receiver_associations(
            srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        srs = np.array(srs_)
        return srs

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
        velv_ = np.asfortranarray(velv, dtype=np.float32)

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

        velv_ = np.asfortranarray(
            np.zeros([nvz_.value + 2, nvx_.value + 2]), dtype=np.float32
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

    def allocate_result_arrays(self):
        self.libfm2dss.allocate_result_arrays()

    def deallocate_result_arrays(self):
        self.libfm2dss.deallocate_result_arrays()

    def get_traveltimes(self):
        nttimes_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_traveltimes(ctypes.byref(nttimes_))
        nttimes = nttimes_.value

        ttimes_ = np.asfortranarray(np.zeros(nttimes), dtype=np.float32)

        tids_ = np.asfortranarray(np.zeros(nttimes), dtype=np.int32)

        self.libfm2dss.get_traveltimes(
            ttimes_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            tids_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        ttimes = np.array(ttimes_)

        return ttimes

    def get_frechet_derivatives(self):

        frechet_nnz_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_frechet_derivatives(ctypes.byref(frechet_nnz_))
        frechet_nnz = frechet_nnz_.value

        frechet_irow_ = np.asfortranarray(
            np.zeros(frechet_nnz), dtype=np.int32
        )
        frechet_icol_ = np.asfortranarray(
            np.zeros(frechet_nnz), dtype=np.int32
        )
        frechet_val_ = np.asfortranarray(
            np.zeros(frechet_nnz), dtype=np.float32
        )

        self.libfm2dss.get_frechet_derivatives(
            frechet_irow_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            frechet_icol_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            frechet_val_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        jrow = np.array(frechet_irow_) - 1
        jcol = np.array(frechet_icol_) - 1
        jval = np.array(frechet_val_)

        nvx_ = ctypes.c_int(-99)
        nvz_ = ctypes.c_int(-99)

        self.libfm2dss.get_number_of_velocity_model_vertices(
            ctypes.byref(nvx_), ctypes.byref(nvz_)
        )
        nvx = nvx_.value
        nvz = nvz_.value

        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc = nsrc_.value
        nrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_receivers(ctypes.byref(nrc_))
        nrc = nrc_.value

        return scipy.sparse.csr_array(
            (jval, (jrow, jcol)), shape=(nsrc * nrc, (nvx + 2) * (nvz + 2))
        )

    def get_raypaths(self):

        npaths_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_raypaths(ctypes.byref(npaths_))
        npaths = int(npaths_.value)

        max_nppts_ = ctypes.c_int(-99)
        self.libfm2dss.get_maximum_number_of_points_per_raypath(
            ctypes.byref(max_nppts_)
        )
        max_nppts = int(max_nppts_.value)

        paths_ = np.asfortranarray(
            np.zeros([npaths, max_nppts, 2]), dtype=np.float32
        )

        nppts_ = np.asfortranarray(np.zeros([npaths]), dtype=np.int32)

        self.libfm2dss.get_raypaths(
            paths_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nppts_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        paths = []

        for i in range(npaths):
            lat = paths_[i, 0 : nppts_[i], 0]
            long = paths_[i, 0 : nppts_[i], 1]
            path = np.array([long, lat]).T
            paths.append(path)

        return paths

    def get_traveltime_fields(self):
        nsrc_ = ctypes.c_int(-99)
        self.libfm2dss.get_number_of_sources(ctypes.byref(nsrc_))
        nsrc = nsrc_.value

        nnx_ = ctypes.c_int(-99)
        nnz_ = ctypes.c_int(-99)

        self.libfm2dss.get_number_of_grid_nodes(ctypes.byref(nnx_), ctypes.byref(nnz_))
        nnx = nnx_.value
        nnz = nnz_.value

        tfields_ = np.asfortranarray(
            np.zeros([nsrc, nnz, nnx]), dtype=np.float32
        )

        self.libfm2dss.get_traveltime_fields(
            tfields_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        tfields = np.array(tfields_)
        return tfields
