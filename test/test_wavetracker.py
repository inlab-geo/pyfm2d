#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:33:05 2025

@author: malcolm
"""
import numpy as np

from pyfm2d import WaveTrackerOptions, display_model, BasisModel
from pyfm2d.wavetracker import _calc_wavefronts_process, _calc_wavefronts_multithreading

PLOT = True


def get_sources():
    return np.array([[0.1, 0.15]])


def get_receivers():
    return np.array([[0.8, 1], [1.0, 0.6]])


def create_velocity_grid_model():
    m = np.array(
        [
            [1, 1.1, 1.1, 1.0],
            [1.0, 1.2, 1.4, 1.3],
            [1.1, 1.2, 1.3, 1.2],
            [1.1, 1.1, 1.2, 1.2],
        ]
    )
    g = BasisModel(m)
    mp = g.get_velocity()
    mp[1, 1] = 0.7
    mp[2, 2] = 0.9
    mp[2, 1] = 1.3
    g.set_velocity(mp)
    return g


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

    if PLOT:
        display_model(g.get_velocity(), paths=result.paths)


def test_calc_wavefonts_multithreading():
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = np.concatenate([get_sources()] * 4)

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
