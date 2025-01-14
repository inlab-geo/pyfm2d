#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:33:05 2025

@author: malcolm
"""
import numpy as np
from scipy.stats import multivariate_normal

# import matplotlib.pyplot as plt
import pyfm2d.wavetracker as wt


def get_gauss_model(
    extent, nx, ny, factor=1.0
):  # build two gaussian anomaly velocity model
    vc1 = 1700.0 * factor  # velocity of circle 1
    vc2 = 2300.0 * factor  # velocity of circle 2
    vb = 2000.0 * factor  # background velocity
    dx = (extent[1] - extent[0]) / nx  # cell width
    dy = (extent[3] - extent[2]) / ny  # cell height
    xc = np.linspace(extent[0], extent[1], nx)  # cell centre
    yc = np.linspace(extent[2], extent[3], ny)  # cell centre
    X, Y = np.meshgrid(xc, yc, indexing="ij")  # cell centre mesh

    # Multivariate Normal
    dex = extent[1] - extent[0]
    dey = extent[3] - extent[2]
    c1x = extent[0] + (7.0 - extent[0]) * dex / 20.0
    c2x = extent[0] + (12.0 - extent[0]) * dex / 20.0
    c1y = extent[0] + (22.0 - extent[0]) * dey / 30.0
    c2y = extent[0] + (10.0 - extent[0]) * dey / 30.0
    s1 = 6.0 * dex / 20.0
    s2 = 10.0 * dex / 20.0
    c1, sig1 = np.array([c1x, c1y]) * factor, s1 * (
        factor**2
    )  # location and radius of centre of first circle
    c2, sig2 = np.array([c2x, c2y]) * factor, s2 * (
        factor**2
    )  # location and radius of centre of first circle
    rv1 = multivariate_normal(c1, [[sig1, 0], [0, sig1]])
    rv2 = multivariate_normal(c2, [[sig2, 0], [0, sig2]])

    # Probability Density
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    gauss1, gauss2 = rv1.pdf(pos), rv2.pdf(pos)
    return (
        vb * np.ones([nx, ny])
        + (vc1 - vb) * gauss1 / np.max(gauss1)
        + (vc2 - vb) * gauss2 / np.max(gauss2)
    )


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
