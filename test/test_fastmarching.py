import numpy as np
import faulthandler

from pyfm2d import fastmarching as fmm



def test_set_solver_options():
    paths = True
    frechet = True
    times = True
    tfieldsource = -1
    tfieldsource = 0
    sourcegridrefine = True
    sourcedicelevel = 5
    sourcegridsize = 10
    earthradius = 6371.0
    schemeorder = 1
    nbsize = 0.5
    degrees = False
    velocityderiv = False
    # extent=[0.,1.,0.,1.]
    dicex = 8
    dicey = 8
    lpaths, lttimes, lfrechet = 0, 0, 0
    if paths:
        lpaths = -1
    if times:
        lttimes = 1
    if frechet:
        lfrechet = 1

    fmm.set_solver_options(
        np.int32(dicex),
        np.int32(dicey),
        np.int32(sourcegridrefine),
        np.int32(sourcedicelevel),
        np.int32(sourcegridsize),
        np.float32(earthradius),
        np.int32(schemeorder),
        np.float32(nbsize),
        np.int32(lttimes),
        np.int32(lfrechet),
        np.int32(tfieldsource + 1),
        np.int32(lpaths),
    )

    gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf = (
        fmm.get_solver_options()
    )

    assert gdx == dicex
    assert gdz == dicey
    assert asgr == sourcegridrefine
    assert sgdl == sourcedicelevel
    assert sgs == sourcegridsize
    assert earth == earthradius
    assert fom == schemeorder
    assert snb == nbsize
    assert fsrt == lttimes
    assert cfd == lfrechet
    assert wttf == tfieldsource + 1
    assert wrgf == lpaths


def test_fmmin2d():
    fmm.fmmin2d()


def test_read_solver_options():
    fmm.read_solver_options("fm2dss.in")
    gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf = (
        fmm.get_solver_options()
    )
    # read file manually and compare


def test_read_velocity_model():
    fmm.read_velocity_model("gridc.vtx")
    nvx, nvz, goxd, gozd, dvxd, dvzd, velv = fmm.get_velocity_model()
    # read file manually and compare


def test_set_velocity_model():
    # create a velocity model e.g. checkerboard
    # fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)
    # check that fmm.get_velocity_model() returns the same values
    pass


def test_read_sources():
    fmm.read_sources("sources.dat")
    scx, scz = fmm.get_sources()
    # read file manually and compare


def test_read_receivers():
    fmm.read_receivers("receivers.dat")
    rcx, rcz = fmm.get_receivers()
    # read file manually and compare


def test_read_source_receiver_associations():
    fmm.read_source_receiver_associations("otimes.dat")
    srs = fmm.get_source_receiver_associations()
    # read file manually and compare


def test_allocate_result_arrays():
    fmm.allocate_result_arrays()


def test_track():
    fmm.track()


def test_get_traveltimes():

    ttimes = fmm.get_traveltimes()
    print(ttimes[0:5])


def test_get_raypaths():
    paths = fmm.get_raypaths()
    print(paths[0][0:10, :])


def test_get_traveltime_fields():
    tfields = fmm.get_traveltime_fields()
    print(tfields[0, 0:5, 0:5])


def test_get_frechet_derivatives():
    frechet = fmm.get_frechet_derivatives()
    print(frechet)


def test_deallocate_result_arrays():
    fmm.deallocate_result_arrays()
