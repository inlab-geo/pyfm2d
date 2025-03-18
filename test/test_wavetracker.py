import numpy as np
import pytest
from pyfm2d import WaveTrackerOptions, display_model, BasisModel
from pyfm2d.wavetracker import (
    _calc_wavefronts_process,
    _calc_wavefronts_multithreading,
    cleanup,
    _build_velocity_grid,
)
import pyfm2d.fastmarching as fmm

PLOT = False
HOMOGENOUS_VELOCITY = 2.0


def get_sources():
    # to ensure separation of sources and receivers
    # get some random sources in one quadrant
    return np.random.uniform(0.05, 0.45, (5, 2))


def get_receivers():
    # to ensure separation of sources and receivers
    # get some random receivers in the opposite quadrant
    return np.random.uniform(0.55, 0.95, (3, 2))


def create_velocity_grid_model():
    # Create a simple velocity model that we can easily
    # manually calculate the travel times for.
    # In this case, we have a 5x5 grid with a velocity of 2.0
    # so the travel time from the source to the receiver should be
    # 0.5 times the path length.
    m = np.ones((5, 5)) * HOMOGENOUS_VELOCITY
    g = BasisModel(m)
    return g


def calculate_expected_tt(src, rec):
    diff = (src[:, np.newaxis] - rec).reshape(-1, 2)  # some broadcasting magic
    return np.sqrt(np.sum(diff**2, axis=1)) / HOMOGENOUS_VELOCITY


def test__calc_wavefonts_process():
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = get_sources()

    options = WaveTrackerOptions(times=True, paths=True, frechet=True)
    result = _calc_wavefronts_process(
        g.get_velocity(),
        recs,
        srcs,
        options=options,
    )

    assert result.ttimes is not None
    assert result.paths is not None
    assert result.frechet is not None

    # Check the travel times
    expected_tt = calculate_expected_tt(srcs, recs)

    # fmm seems quite inaccurate because of the small grid size
    # and putting float32 everywhere
    assert np.allclose(result.ttimes, expected_tt, atol=1e-2)

    if PLOT:
        display_model(g.get_velocity(), paths=result.paths)


def test_cleanup():

    @cleanup
    def failing_function():
        # setup required to allocate arrays in the first place
        v = create_velocity_grid_model().get_velocity()
        recs = get_receivers()
        srcs = get_sources()
        options = WaveTrackerOptions(times=True, paths=True, frechet=True)
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
            options.tsource,
            options.lpaths,
        )

        fmm.set_sources(srcs[:, 1], srcs[:, 0])  # ordering inherited from fm2dss.f90
        fmm.set_receivers(recs[:, 1], recs[:, 0])  # ordering inherited from fm2dss.f90

        nvx, nvy = v.shape
        # grid node spacing in lat and long
        extent = [0, 1, 0, 1]
        dlat = (extent[3] - extent[2]) / (nvy - 1)
        dlong = (extent[1] - extent[0]) / (nvx - 1)

        vc = _build_velocity_grid(v)

        fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)

        # set up time calculation between all sources and receivers
        associations = np.ones((recs.shape[0], srcs.shape[0]))
        fmm.set_source_receiver_associations(associations)

        fmm.allocate_result_arrays()
        raise ValueError("This function should fail")

    with pytest.raises(ValueError):
        failing_function()

    # If the clean up worked, we should now be able to
    # allocate again without any issues
    fmm.allocate_result_arrays()
    # Now cleanup again to avoid any issues with the next test
    fmm.deallocate_result_arrays()


def test_calc_wavefonts_multithreading():
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = np.concatenate([get_sources() for _ in range(4)])

    options = WaveTrackerOptions(times=True, paths=True, frechet=True)
    result = _calc_wavefronts_multithreading(
        g.get_velocity(),
        recs,
        srcs,
        options=options,
        nthreads=4,
    )

    assert result.ttimes is not None
    assert result.paths is not None
    assert result.frechet is not None

    # Check the travel times
    expected_tt = calculate_expected_tt(srcs, recs)
    assert np.allclose(result.ttimes, expected_tt, atol=1e-2)

    if PLOT:
        display_model(g.get_velocity(), paths=result.paths)
