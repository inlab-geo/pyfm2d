#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:33:05 2025

@author: malcolm
"""
import numpy as np

# import matplotlib.pyplot as plt
import pyfm2dss.wavetracker as wt

# import time
# from tqdm import tqdm
import sys

# Simple test routine for wavetracker class and its use of low level class functions in pyfm2dss


def build_velocitygrid(
    v, extent
):  # add cushion nodes about velocity model to be compatible with fm2dss.f90 input
    #
    # here extent[3],extent[2] is N-S range of grid nodes
    #      extent[0],extent[1] is W-E range of grid nodes
    nx, ny = v.shape
    dlat, dlong = (extent[3] - extent[2]) / (ny - 1), (extent[1] - extent[0]) / (
        nx - 1
    )  # grid node spacing in lat and long

    # gridc.vtx requires a single cushion layer of nodes surrounding the velocty model
    # build velocity model with cushion velocities

    noncushion = np.zeros(
        (nx + 2, ny + 2), dtype=bool
    )  # bool array to identify cushion and non cushion nodes
    noncushion[1 : nx + 1, 1 : ny + 1] = True

    # mapping from cushion indices to non cushion indices
    nodemap = np.zeros((nx + 2, ny + 2), dtype=int)
    nodemap[1 : nx + 1, 1 : ny + 1] = np.array(range((nx * ny))).reshape((nx, ny))
    nodemap = nodemap[:, ::-1]

    # build velocity nodes
    # additional boundary layer of velocities are duplicates of the nearest actual velocity value.
    vc = np.ones((nx + 2, ny + 2))
    vc[1 : nx + 1, 1 : ny + 1] = v
    vc[1 : nx + 1, 0] = v[:, 0]  # add velocities in the cushion boundary layer
    vc[1 : nx + 1, -1] = v[:, -1]  # add velocities in the cushion boundary layer
    vc[0, 1 : ny + 1] = v[0, :]  # add velocities in the cushion boundary layer
    vc[-1, 1 : ny + 1] = v[-1, :]  # add velocities in the cushion boundary layer
    vc[0, 0], vc[0, -1], vc[-1, 0], vc[-1, -1] = v[0, 0], v[0, -1], v[-1, 0], v[-1, -1]
    vc = vc[:, ::-1]

    return nx, ny, dlat, dlong, vc, noncushion, nodemap.flatten()


m = np.array(
    [
        [1, 1.1, 1.1, 1.0],
        [1.0, 1.2, 1.4, 1.3],
        [1.1, 1.2, 1.3, 1.2],
        [1.1, 1.1, 1.2, 1.2],
    ]
)

g = wt.GridModel(m)
mp = g.getVelocity()
mp[1, 1] = 0.7
mp[2, 2] = 0.9
mp[2, 1] = 1.3
g.setVelocity(mp)

# -------------------------------------------------------

srcs = np.array([0.1, 0.15])  # source (x,y)

recs = np.array([[0.8, 1], [1.0, 0.6]])  # receivers [(x1,y1),(x2,y2)]

# -------------------------------------------------------
from scipy.stats import multivariate_normal


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


factor = 1.0
extent = [0.0, 20.0 * factor, 0.0, 30.0 * factor]
# m=get_spherical_model(extent,32,48)
m = get_gauss_model(extent, 32, 48)
g = wt.BasisModel(m, extent=extent)
recs = g.generateSurfacePoints(
    10, extent=extent, surface=[False, True, False, False], addCorners=False
)  # generate receivers around edge
srcs = g.generateSurfacePoints(
    10, extent=extent, surface=[True, False, False, False], addCorners=False
)  # generate receivers around edge

# -------------------------------------------------------
v = g.getVelocity()

# run wave front tracker
myfmm = wt.WaveTracker()

# test options

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


myfmm.fmm.set_solver_options(
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
    myfmm.fmm.get_solver_options()
)

print(
    "Original options: ",
    dicex,
    dicey,
    int(sourcegridrefine),
    sourcedicelevel,
    sourcegridsize,
    earthradius,
    schemeorder,
    nbsize,
    lttimes,
    lfrechet,
    tfieldsource + 1,
    lpaths,
)
print(
    "Recovered options:",
    gdx,
    gdz,
    asgr,
    sgdl,
    sgs,
    earth,
    fom,
    snb,
    fsrt,
    cfd,
    wttf,
    wrgf,
)

# test velocity model input

nvx, nvy, dlat, dlong, vc, noncushion, nodemap = build_velocitygrid(
    v, extent
)  # add cushion layer to velocity model and get parameters

nvx = np.int32(nvx)
nvy = np.int32(nvy)
extent = np.array(extent, dtype=np.float32)
dlat = np.float32(dlat)
dlong = np.float32(dlong)
vc = vc.astype(np.float32)

myfmm.fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)

nvxo, nvzo, goxd, gozd, dvxd, dvzd, velv = myfmm.fmm.get_velocity_model()

print("Original velocity paramaters:  ", nvx, nvy, extent[3], extent[0], dlat, dlong)
print("Recovered velocity paramaters: ", nvxo, nvzo, goxd, gozd, dvxd, dvzd)
print("Original velocity field: \n", vc)
print("Recovered velocity field: \n", velv)

# test source and receiver

recs = recs.reshape(-1, 2)
srcs = srcs.reshape(-1, 2)

rcy = np.float32(recs[:, 1])
rcx = np.float32(recs[:, 0])

myfmm.fmm.set_receivers(rcy, rcx)  # set receivers

scy = np.float32(srcs[:, 1])
scx = np.float32(srcs[:, 0])

myfmm.fmm.set_sources(scy, scx)  # set sources

scyo, scxo = myfmm.fmm.get_sources()

print("Source\n", "Original", srcs, "\n", "recovered", scxo, scyo)

rcyo, rcxo = myfmm.fmm.get_receivers()

print("Receivers\n", "Original", recs[:, 0], recs[:, 1], "\n", "recovered", rcxo, rcyo)

# test source/receiver associations
srs = np.ones(
    (len(recs), len(srcs)), dtype=np.int32
)  # set up time calculation between all sources and receivers

myfmm.fmm.set_source_receiver_associations(srs)

srso = myfmm.fmm.get_source_receiver_associations()

print("Original associations:\n", srs)
print("Recovered associations:\n", srso)


myfmm.fmm.allocate_result_arrays()  # allocate memory for Fortran arrays

myfmm.fmm.track()
# check results
kms2deg = 180.0 / (earthradius * np.pi)

if times:
    ttimes = myfmm.fmm.get_traveltimes()
    if not degrees:
        ttimes *= kms2deg  # adjust travel times because inputs are not in degrees

if paths:
    raypaths = myfmm.fmm.get_raypaths()

if frechet:
    frechetvals = (
        myfmm.fmm.get_frechet_derivatives()
    )  # THIS IS PROBABLY LACKS ADJUSTMENT FOR CUSHION NODES see routine read_fmst_frechet in _core.py
    if not degrees:
        frechetvals *= kms2deg  # adjust travel times because inputs are not in degrees
    # if(not velocityderiv):  THIS ONLY WORKS WHEN frechetvals has cushion removed otherwise they are not of teh same shape
    #    x2 = -(v*v).reshape(-1)
    #    frechetvals = frechetvals.multiply(x2)

if tfieldsource >= 0:
    tfieldvals = myfmm.fmm.get_traveltime_fields()
    if not degrees:
        tfieldvals *= kms2deg  # adjust travel times because inputs are not in degrees

myfmm.fmm.deallocate_result_arrays()

sys.exit()


fmm.calc_wavefronts(g.getVelocity(), recs, srcs, verbose=True, frechet=True, paths=True)
print(" Number of paths calculated = ", len(fmm.paths))
print(" Number of travel times calculated = ", len(fmm.ttimes))
print(" Shape of frechet matrix = ", fmm.frechet.shape)
