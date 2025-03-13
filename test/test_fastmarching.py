import os
import numpy as np
import pytest
from dataclasses import dataclass

from pathlib import Path

from pyfm2d import fastmarching as fmm

DATADIR = Path(__file__).parent
FMINFILE = str(DATADIR / "fm2dss.in")
VELMODELFILE = str(DATADIR / "gridc.vtx")
SOURCESFILE = str(DATADIR / "sources.dat")
RECEIVERSFILE = str(DATADIR / "receivers.dat")
ASSOCIATIONSFILE = str(DATADIR / "otimes.dat")


@dataclass
class FMMOptions:
    dicex: int = 8
    dicey: int = 8
    sourcegridrefine: int = 1
    sourcedicelevel: int = 5
    sourcegridsize: int = 10
    earthradius: float = 6371.0
    schemeorder: int = 1
    nbsize: float = 0.5
    lttimes: int = 1
    lfrechet: int = 1
    tfieldsource: int = 0
    lpaths: int = -1


def test_set_solver_options():
    options = FMMOptions()

    fmm.set_solver_options(
        options.dicex,
        options.dicey,
        options.sourcegridrefine,
        options.sourcedicelevel,
        options.sourcegridsize,
        options.earthradius,
        options.schemeorder,
        options.nbsize,
        options.lttimes,
        options.lfrechet,
        options.tfieldsource + 1,
        options.lpaths,
    )

    gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf = (
        fmm.get_solver_options()
    )

    assert gdx == options.dicex
    assert gdz == options.dicey
    assert asgr == options.sourcegridrefine
    assert sgdl == options.sourcedicelevel
    assert sgs == options.sourcegridsize
    assert earth == options.earthradius
    assert fom == options.schemeorder
    assert snb == options.nbsize
    assert fsrt == options.lttimes
    assert cfd == options.lfrechet
    assert wttf == options.tfieldsource + 1
    assert wrgf == options.lpaths


def test_fmmin2d():
    os.chdir(Path(__file__).parent)
    fmm.fmmin2d()
    os.chdir(Path(__file__).parent.parent)


def test_read_solver_options():
    fmm.read_solver_options(FMINFILE)
    gdx, gdz, asgr, sgdl, sgs, earth, fom, snb, fsrt, cfd, wttf, wrgf = (
        fmm.get_solver_options()
    )

    # read file manually and compare
    with open(FMINFILE, "r") as f:
        lines = f.readlines()

    assert gdx == int(lines[7].split()[0])
    assert gdz == int(lines[7].split()[1])

    assert asgr == int(lines[8].split()[0])

    assert sgdl == int(lines[9].split()[0])
    assert sgs == int(lines[9].split()[1])

    assert earth == float(lines[10].split()[0])

    assert fom == int(lines[11].split()[0])

    assert snb == float(lines[12].split()[0])

    assert fsrt == int(lines[16].split()[0])

    assert cfd == int(lines[18].split()[0])

    assert wttf == int(lines[20].split()[0])

    assert wrgf == int(lines[22].split()[0])


def test_read_velocity_model():
    fmm.read_velocity_model(VELMODELFILE)
    nvx, nvz, goxd, gozd, dvxd, dvzd, velv = fmm.get_velocity_model()

    # read file manually and compare
    with open(VELMODELFILE, "r") as f:
        lines = f.readlines()

    assert nvx == int(lines[0].split()[0])
    assert nvz == int(lines[0].split()[1])

    assert goxd == pytest.approx(float(lines[1].split()[0]))
    assert gozd == pytest.approx(float(lines[1].split()[1]))

    assert dvxd == pytest.approx(float(lines[2].split()[0]))
    assert dvzd == pytest.approx(float(lines[2].split()[1]))

    # The true model file has 2 columns where the velocity is stored
    # However, the Fortran code only reads the first column
    # DO i = 0, nvz + 1
    #     DO j = 0, nvx + 1
    #         READ (10, *) velv(i, j)
    #     END DO
    # END DO
    assert np.array_equal(
        velv.flatten(), np.loadtxt(VELMODELFILE, skiprows=3, usecols=0)
    )


def test_set_velocity_model():
    # fmm.set_velocity_model takes the cushionned velocity array
    # i.e. it expects something of shape (nvy + 2, nvx + 2)
    nvy, nvx = 10, 10
    extent = [1.0, 2.0, 3.0, 4.0]
    dlat, dlong = 0.1, 0.1
    vc = np.full((nvy + 2, nvx + 2), 2000.0)

    fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)
    nx, nz, goxd, gozd, dvxd, dvzd, velv = fmm.get_velocity_model()

    assert nx == nvy
    assert nz == nvx
    assert goxd == pytest.approx(extent[3])
    assert gozd == pytest.approx(extent[0])
    assert dvxd == pytest.approx(dlong)
    assert dvzd == pytest.approx(dlat)

    assert np.allclose(velv, vc, atol=1e-7)
    assert velv.shape == (nvy + 2, nvx + 2)


def test_read_sources():
    fmm.read_sources(SOURCESFILE)
    scx, scz = fmm.get_sources()
    sources = np.array([scx, scz]).T

    assert np.array_equal(sources, np.loadtxt(SOURCESFILE, skiprows=1))


def test_set_sources():
    sources = np.array([[0.1, 0.15], [0.2, 0.25]])
    scx = sources[:, 0]
    scy = sources[:, 1]
    fmm.set_sources(scy, scx)  # note the order of scx and scy
    _scx, _scz = fmm.get_sources()
    assert np.allclose(_scx, scy, atol=1e-7)
    assert np.allclose(_scz, scx, atol=1e-7)


def test_read_receivers():
    fmm.read_receivers(RECEIVERSFILE)
    rcx, rcz = fmm.get_receivers()
    receviers = np.array([rcx, rcz]).T

    assert np.array_equal(receviers, np.loadtxt(RECEIVERSFILE, skiprows=1))


def test_set_receivers():
    receivers = np.array([[0.8, 1], [1.0, 0.6]])
    rcx = receivers[:, 0]
    rcy = receivers[:, 1]
    fmm.set_receivers(rcy, rcx)  # note the order of rcx and rcy
    _rcx, _rcz = fmm.get_receivers()
    assert np.allclose(_rcx, rcy, atol=1e-7)
    assert np.allclose(_rcz, rcx, atol=1e-7)


def test_read_source_receiver_associations():
    fmm.read_sources(SOURCESFILE)
    fmm.read_receivers(RECEIVERSFILE)

    fmm.read_source_receiver_associations(ASSOCIATIONSFILE)
    srs = fmm.get_source_receiver_associations()

    # read file manually and compare
    with open(SOURCESFILE, "r") as f:
        lines = f.readlines()
        nsrc = int(lines[0].split()[0])
    with open(RECEIVERSFILE, "r") as f:
        lines = f.readlines()
        nrec = int(lines[0].split()[0])

    # have to manipulate the array to match the Fortran array
    associations = np.loadtxt(ASSOCIATIONSFILE, dtype=int).reshape(nsrc, nrec).T

    assert srs.shape == (nrec, nsrc)
    assert np.array_equal(srs, associations)


def test_track():
    # track is the main function
    #
    # get_raypaths, get_traveltimes, get_traveltime_fields, get_frechet_derivatives
    # only return values after track has been called
    options = FMMOptions()

    fmm.set_solver_options(
        np.int32(options.dicex),
        np.int32(options.dicey),
        np.int32(options.sourcegridrefine),
        np.int32(options.sourcedicelevel),
        np.int32(options.sourcegridsize),
        np.float32(options.earthradius),
        np.int32(options.schemeorder),
        np.float32(options.nbsize),
        np.int32(options.lttimes),
        np.int32(options.lfrechet),
        np.int32(options.tfieldsource + 1),
        np.int32(options.lpaths),
    )

    fmm.read_velocity_model(VELMODELFILE)
    fmm.read_sources(SOURCESFILE)
    fmm.read_receivers(RECEIVERSFILE)
    fmm.read_source_receiver_associations(ASSOCIATIONSFILE)

    fmm.allocate_result_arrays()

    fmm.track()

    ttimes = fmm.get_traveltimes()
    assert ttimes is not None

    paths = fmm.get_raypaths()
    assert paths is not None

    tfields = fmm.get_traveltime_fields()  ## failing silently here
    assert tfields is not None

    frechet = fmm.get_frechet_derivatives()
    assert frechet is not None

    fmm.deallocate_result_arrays()
