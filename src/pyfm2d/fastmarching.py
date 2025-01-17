import numpy as np
import scipy
import ctypes
from pathlib import Path
from site import getsitepackages


package_dir = Path(getsitepackages()[0]) / "pyfm2d"
lib_path = next(package_dir.glob("libfm2d*.so"))
libfm2d = ctypes.cdll.LoadLibrary(str(lib_path))


def fmmin2d():
    libfm2d.fmmin2d()


def track():
    libfm2d.track()


def read_solver_options(fn_):
    fn = ctypes.c_char_p(fn_.encode("UTF-8"))
    libfm2d.read_configuration(fn, ctypes.c_int(len(fn.value)))


def set_solver_options(
    gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf
):
    # Not sure if the np.int32 and np.float32 are necessary
    # e.g. ctypes.c_int is an alias for ctypes.c_long anyway on 64-bit systems
    # so casting to 32-bit first seems unnecessary
    #
    # unit test passed on my system without the casting
    gdx_ = ctypes.c_int(np.int32(gdx))
    gdz_ = ctypes.c_int(np.int32(gdz))
    asgr_ = ctypes.c_int(np.int32(asgr))
    sgdl_ = ctypes.c_int(np.int32(sgdl))
    sgs_ = ctypes.c_int(np.int32(sgs))
    earth_ = ctypes.c_float(np.float32(earth))
    fom_ = ctypes.c_int(np.int32(fom))
    snb_ = ctypes.c_float(np.float32(snb))

    fsrt_ = ctypes.c_int(np.int32(fsrt))
    cfd_ = ctypes.c_int(np.int32(cfd))
    wttf_ = ctypes.c_int(np.int32(wttf))
    wrgf_ = ctypes.c_int(np.int32(wrgf))

    libfm2d.set_solver_options(
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


def get_solver_options():
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

    libfm2d.get_solver_options(
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


def read_sources(fn_):
    fn = ctypes.c_char_p(fn_.encode("UTF-8"))
    libfm2d.read_sources(fn, ctypes.c_int(len(fn.value)))


def set_sources(scx_, scz_):
    # testing showed that dtype must be np.float32
    # not sure why that's the case here but not for the options
    # must be something to do with the memory allocation in the
    # Fortran code
    scx_ = np.asfortranarray(scx_, dtype=np.float32)
    scz_ = np.asfortranarray(scz_, dtype=np.float32)
    nsrc_ = ctypes.c_int(scx_.shape[0])

    libfm2d.set_sources(
        scx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(nsrc_),
    )


def get_sources():
    nsrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_sources(ctypes.byref(nsrc_))
    nsrc = nsrc_.value
    scx_ = np.empty((nsrc), dtype=ctypes.c_float)
    scz_ = np.empty((nsrc), dtype=ctypes.c_float)
    libfm2d.get_sources(
        scx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(nsrc_),
    )
    scx = np.array(scx_)
    scz = np.array(scz_)
    return scx, scz


def read_receivers(fn_):
    fn = ctypes.c_char_p(fn_.encode("UTF-8"))
    libfm2d.read_receivers(fn, ctypes.c_int(len(fn.value)))


def set_receivers(rcx_, rcz_):
    rcx_ = np.asfortranarray(rcx_, dtype=np.float32)
    rcz_ = np.asfortranarray(rcz_, dtype=np.float32)
    nrc_ = ctypes.c_int(rcx_.shape[0])

    libfm2d.set_receivers(
        rcx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rcz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(nrc_),
    )


def get_receivers():
    nrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_receivers(ctypes.byref(nrc_))
    nrc = nrc_.value
    rcx_ = np.empty((nrc), dtype=ctypes.c_float)
    rcz_ = np.empty((nrc), dtype=ctypes.c_float)
    libfm2d.get_receivers(
        rcx_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rcz_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(nrc_),
    )
    rcx = np.array(rcx_)
    rcz = np.array(rcz_)
    return rcx, rcz


def read_source_receiver_associations(fn_):
    fn = ctypes.c_char_p(fn_.encode("UTF-8"))
    libfm2d.read_source_receiver_associations(fn, ctypes.c_int(len(fn.value)))


def set_source_receiver_associations(srs):
    nsrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_sources(ctypes.byref(nsrc_))
    nrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_receivers(ctypes.byref(nrc_))

    srs_ = np.asfortranarray(srs, dtype=np.int32)
    srs_ = libfm2d.set_source_receiver_associations(
        srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    return


def get_source_receiver_associations():

    nsrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_sources(ctypes.byref(nsrc_))
    nsrc = nsrc_.value

    nrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_receivers(ctypes.byref(nrc_))
    nrc = nrc_.value

    srs_ = np.asfortranarray(np.zeros([nrc, nsrc]), dtype=np.int32)
    libfm2d.get_source_receiver_associations(
        srs_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    srs = np.array(srs_)
    return srs


def read_velocity_model(fn_):
    fn = ctypes.c_char_p(fn_.encode("UTF-8"))
    libfm2d.read_velocity_model(fn, ctypes.c_int(len(fn.value)))


def set_velocity_model(nvx, nvz, goxd, gozd, dvxd, dvzd, velv):
    nvx_ = ctypes.c_int(np.int32(nvx))
    nvz_ = ctypes.c_int(np.int32(nvz))
    goxd_ = ctypes.c_float(np.float32(goxd))
    gozd_ = ctypes.c_float(np.float32(gozd))
    dvxd_ = ctypes.c_float(np.float32(dvxd))
    dvzd_ = ctypes.c_float(np.float32(dvzd))
    velv_ = np.asfortranarray(velv, dtype=np.float32)

    libfm2d.set_velocity_model(
        ctypes.byref(nvx_),
        ctypes.byref(nvz_),
        ctypes.byref(goxd_),
        ctypes.byref(gozd_),
        ctypes.byref(dvxd_),
        ctypes.byref(dvzd_),
        velv_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )


def get_velocity_model():
    nvx_ = ctypes.c_int(-99)
    nvz_ = ctypes.c_int(-99)

    libfm2d.get_number_of_velocity_model_vertices(
        ctypes.byref(nvx_), ctypes.byref(nvz_)
    )

    goxd_ = ctypes.c_float(-99.9)
    gozd_ = ctypes.c_float(-99.9)

    dvxd_ = ctypes.c_float(-99.9)
    dvzd_ = ctypes.c_float(-99.9)

    velv_ = np.asfortranarray(
        np.zeros([nvz_.value + 2, nvx_.value + 2]), dtype=np.float32
    )

    libfm2d.get_velocity_model(
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


def allocate_result_arrays():
    libfm2d.allocate_result_arrays()


def deallocate_result_arrays():
    libfm2d.deallocate_result_arrays()


def get_traveltimes():
    nttimes_ = ctypes.c_int(-99)
    libfm2d.get_number_of_traveltimes(ctypes.byref(nttimes_))
    nttimes = nttimes_.value

    ttimes_ = np.asfortranarray(np.zeros(nttimes), dtype=np.float32)

    tids_ = np.asfortranarray(np.zeros(nttimes), dtype=np.int32)

    libfm2d.get_traveltimes(
        ttimes_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tids_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    ttimes = np.array(ttimes_)

    return ttimes


def get_frechet_derivatives():

    frechet_nnz_ = ctypes.c_int(-99)
    libfm2d.get_number_of_frechet_derivatives(ctypes.byref(frechet_nnz_))
    frechet_nnz = frechet_nnz_.value

    frechet_irow_ = np.asfortranarray(np.zeros(frechet_nnz), dtype=np.int32)
    frechet_icol_ = np.asfortranarray(np.zeros(frechet_nnz), dtype=np.int32)
    frechet_val_ = np.asfortranarray(np.zeros(frechet_nnz), dtype=np.float32)

    libfm2d.get_frechet_derivatives(
        frechet_irow_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        frechet_icol_.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        frechet_val_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    jrow = np.array(frechet_irow_) - 1
    jcol = np.array(frechet_icol_) - 1
    jval = np.array(frechet_val_)

    nvx_ = ctypes.c_int(-99)
    nvz_ = ctypes.c_int(-99)

    libfm2d.get_number_of_velocity_model_vertices(
        ctypes.byref(nvx_), ctypes.byref(nvz_)
    )
    nvx = nvx_.value
    nvz = nvz_.value

    nsrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_sources(ctypes.byref(nsrc_))
    nsrc = nsrc_.value
    nrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_receivers(ctypes.byref(nrc_))
    nrc = nrc_.value

    return scipy.sparse.csr_array(
        (jval, (jrow, jcol)), shape=(nsrc * nrc, (nvx + 2) * (nvz + 2))
    )


def get_raypaths():

    npaths_ = ctypes.c_int(-99)
    libfm2d.get_number_of_raypaths(ctypes.byref(npaths_))
    npaths = int(npaths_.value)

    max_nppts_ = ctypes.c_int(-99)
    libfm2d.get_maximum_number_of_points_per_raypath(ctypes.byref(max_nppts_))
    max_nppts = int(max_nppts_.value)

    paths_ = np.asfortranarray(np.zeros([npaths, max_nppts, 2]), dtype=np.float32)

    nppts_ = np.asfortranarray(np.zeros([npaths]), dtype=np.int32)

    libfm2d.get_raypaths(
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


def get_traveltime_fields():
    nsrc_ = ctypes.c_int(-99)
    libfm2d.get_number_of_sources(ctypes.byref(nsrc_))
    nsrc = nsrc_.value

    nnx_ = ctypes.c_int(-99)
    nnz_ = ctypes.c_int(-99)

    libfm2d.get_number_of_grid_nodes(ctypes.byref(nnx_), ctypes.byref(nnz_))
    nnx = nnx_.value
    nnz = nnz_.value

    tfields_ = np.asfortranarray(np.zeros([nsrc, nnz, nnx]), dtype=np.float32)

    libfm2d.get_traveltime_fields(
        tfields_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    tfields = np.array(tfields_)
    return tfields
