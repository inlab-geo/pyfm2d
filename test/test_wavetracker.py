#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:33:05 2025

@author: malcolm
"""
import numpy as np

import pyfm2d.wavetracker as wt


PLOT = False


def get_sources():
    return np.array([0.1, 0.15])


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
    g = wt.BasisModel(m)
    mp = g.get_velocity()
    mp[1, 1] = 0.7
    mp[2, 2] = 0.9
    mp[2, 1] = 1.3
    g.set_velocity(mp)
    return g


def test_calc_wavefonts():
    g = create_velocity_grid_model()
    recs = get_receivers()
    srcs = get_sources()

    wavetracker = wt.WaveTracker()
    wavetracker.calc_wavefronts(
        g.get_velocity(),
        recs,
        srcs,
        verbose=True,
        frechet=True,
        paths=True,
    )

    assert wavetracker.ttimes is not None
    assert wavetracker.paths is not None
    assert wavetracker.frechet is not None

    if PLOT:
        wt.display_model(g.get_velocity(), paths=wavetracker.paths)
