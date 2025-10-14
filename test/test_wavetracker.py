import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pyfm2d import WaveTrackerOptions, display_model, BasisModel, calc_wavefronts
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

def convert_kms_2_deg(src,rec,extent):
    degperkms = 180./(6371.0*np.pi)
    rec = degperkms*rec
    src = degperkms*src
    ext = [i*degperkms for i in extent] # unit box in kms converted to degrees
    return src,rec,ext

def test__calc_wavefonts_process():
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = get_sources()
    cartesian = False
    extent = [0., 1., 0., 1.]

    # Check the travel times
    expected_tt = calculate_expected_tt(srcs, recs)

    #srcs,recs,extent = convert_kms_2_deg(srcs,recs,extent) # For Spherical model we should convert the input spatial units to degrees

    options = WaveTrackerOptions(times=True, paths=True, frechet=True, cartesian=True)
    result = _calc_wavefronts_process(
        g.get_velocity(),
        recs,
        srcs,
        extent=extent,
        options=options,
    )

    assert result.ttimes is not None
    assert result.paths is not None
    assert result.frechet is not None

    # fmm seems quite inaccurate because of the small grid size
    # and putting float32 everywhere
    print('expected_tt',expected_tt)
    print('result.ttimes',result.ttimes)
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
            options.lcartesian,
            int(options.quiet),
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
    extent = [0.,1.,0.,1.]

    # Check the travel times
    expected_tt = calculate_expected_tt(srcs, recs)

    srcs,recs,extent = convert_kms_2_deg(srcs,recs,extent) # For Spherical model we should convert the input spatial units to degrees

    options = WaveTrackerOptions(times=True, paths=True, frechet=True)
    result = _calc_wavefronts_multithreading(
        g.get_velocity(),
        recs,
        srcs,
        extent=extent,
        options=options,
        nthreads=4,
    )

    assert result.ttimes is not None
    assert result.paths is not None
    assert result.frechet is not None

    assert np.allclose(result.ttimes, expected_tt, atol=1e-2)

    if PLOT:
        display_model(g.get_velocity(), paths=result.paths)


def test_user_provided_threadpool_executor():
    """Test that ThreadPoolExecutor raises an error due to Fortran shared memory conflicts"""
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = get_sources()
    extent = [0., 1., 0., 1.]
    
    srcs, recs, extent = convert_kms_2_deg(srcs, recs, extent)
    
    options = WaveTrackerOptions(times=True, paths=True, frechet=True)
    
    # ThreadPoolExecutor cannot be used due to shared memory conflicts in the Fortran implementation
    # Multiple threads trying to allocate the same global Fortran arrays causes segfaults
    with pytest.raises(ValueError, match="ThreadPoolExecutor is not supported"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = calc_wavefronts(
                g.get_velocity(),
                recs,
                srcs,
                pool=pool,
                extent=extent,
                options=options,
            )


def test_user_provided_processpool_executor():
    """Test calc_wavefronts with user-provided ProcessPoolExecutor"""
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = get_sources()
    extent = [0., 1., 0., 1.]
    
    # Calculate expected travel times before conversion
    expected_tt = calculate_expected_tt(srcs, recs)
    
    srcs, recs, extent = convert_kms_2_deg(srcs, recs, extent)
    
    options = WaveTrackerOptions(times=True, paths=True, frechet=True)
    
    with ProcessPoolExecutor(max_workers=2) as pool:
        result = calc_wavefronts(
            g.get_velocity(),
            recs,
            srcs,
            pool=pool,
            extent=extent,
            options=options,
        )
    
    assert result.ttimes is not None
    assert result.paths is not None
    assert result.frechet is not None
    
    assert np.allclose(result.ttimes, expected_tt, atol=1e-2)


def test_pool_reuse():
    """Test that a user-provided pool can be reused for multiple calculations"""
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs1 = get_sources()[:2]
    srcs2 = get_sources()[2:4]
    extent = [0., 1., 0., 1.]
    
    srcs1, _, extent1 = convert_kms_2_deg(srcs1, recs, extent)
    srcs2, recs, extent2 = convert_kms_2_deg(srcs2, recs, extent)
    
    options = WaveTrackerOptions(times=True, paths=False, frechet=False)
    
    with ProcessPoolExecutor(max_workers=2) as pool:
        # First calculation
        result1 = calc_wavefronts(
            g.get_velocity(),
            recs,
            srcs1,
            pool=pool,
            extent=extent1,
            options=options,
        )
        
        # Second calculation with same pool
        result2 = calc_wavefronts(
            g.get_velocity(),
            recs,
            srcs2,
            pool=pool,
            extent=extent2,
            options=options,
        )
    
    assert result1.ttimes is not None
    assert result2.ttimes is not None
    assert len(result1.ttimes) == len(srcs1) * len(recs)
    assert len(result2.ttimes) == len(srcs2) * len(recs)


def test_calc_wavefonts_multithreading_vs_serial():
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = get_sources()
    extent = [0.,1.,0.,1.]

    srcs,recs,extent = convert_kms_2_deg(srcs,recs,extent) # For Spherical model we should convert the input spatial units to degrees

    options = WaveTrackerOptions(times=True, paths=True, frechet=True)
    result_serial = _calc_wavefronts_process(
        g.get_velocity(),
        recs,
        srcs,
        extent=extent,
        options=options,
    )

    result_parallel = _calc_wavefronts_multithreading(
        g.get_velocity(),
        recs,
        srcs,
        extent=extent,
        options=options,
        nthreads=4,
    )

    assert np.allclose(result_serial.ttimes, result_parallel.ttimes)
    for i in range(len(result_serial.paths)):
        assert np.allclose(result_serial.paths[i], result_parallel.paths[i]), f"Path {i} is not equal"
    assert np.allclose(result_serial.frechet.toarray(), result_parallel.frechet.toarray())
