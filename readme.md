# pyfm2d

![Python3](https://img.shields.io/badge/python-3.x-brightgreen.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

_2D ray tracing and wavefront tracking_


This repository contains ctypes based Python wrappers for the program `fm2dss.F90` by Nick Rawlinson for 2D ray tracing and wavefront tracking using the Fast Marching method in a 2D seismic velocity or slowness model.


## Installation

```
pip install git+https://github.com/inlab-geo/pyfm2d
```
## Documentation

This is a package of three main user facing classes.

basisModel - which contains functions to define, retrieve and manipulate 2D seismic velocity or slowness models expressed in a choice of model bases.

plot - which contains functions to plot the above

WaveTracker - which implements first arrival travel time calculations between sources and receivers in the 2D model. Optionally returns travel times, ray paths, wavefronts and Frechet derivative matrices.

Here is the docstring of the function `calc_wavefronts()` which does the work:

```
A function to perform 2D Fast Marching of wavefronts from sources in a 2D velocity model.

    Inputs:
        v, ndarray(nx,ny)          : coefficients of velocity field in 2D grid
        recs, ndarray(nr,2)        : receiver locations (x,y). Where nr is the number of receivers.
        srcs, ndarray(ns,2)        : source locations (x,y). Where ns is the number of receivers.
        paths, bool                : raypath option (True=calculate and return ray paths)
        frechet, bool              : frechet derivative option (True=calculate and return frechet derivative matrix for raypths in each cell)
        times, bool                : travel times derivative option (True=calculate and travel times)
        tfieldsource, int          : source id to calculate travel time field (<0=none,>=0=source id)
        sourcegridrefine, bool     : Apply sourcegrid refinement (default=True)
        sourcedicelevel, int       : Source discretization level. Number of sub-divisions per cell (default=5, i.e. 1 model cell becomes 5x5 sub-cells)
        sourcegridsize, int        : Number of model cells to refine about source at sourcedicelevel (default=10, i.e. 10x10 cells are refines about source)
        earthradius, float         : radius of Earth in km, used for spherical to Cartesian transform (default=6371.0)
        schemeorder, int           : switch to use first order (0) or mixed order(1) scheme (default=1,mixed)
        nbsize,float               : Narrow band size (0-1) as fraction of nnx*nnz (default=0.5)
        degrees, bool              : True if input distances are in degrees (default=False). Uses earthradius to convert to km.
        velocityderiv, bool        : Switch to return Frechet derivatives of travel times w.r.t. velocities (True) rather than slownesses (False, default).
        extent, list               : 4-tuple of model extent [xmin,xmax,ymin,ymax]. (default=[0.,1.,0.,1.])
        dicex, int                 : x-subgrid discretization level for B-spline interpolation of input model (default=8)
        dicey, int                 : y-subgrid discretization level for B-spline interpolation of input model (default=8)

    Returns
        WaveTracker.ttimes, ndarray(ns*nr)   : first arrival travel times between ns sources and nr receivers.
        WaveTracker.paths, list(ns*nr)       : list of 2-D arrays (x,y) for all ns*nr raypaths.
        WaveTracker.ttfield, ndarray(mx,my)  : 2-D array of travel time field for source tfieldsource at resolution mx*my
                                               (mx = dicex*(nx-1) + 1, my = dicey*(ny-1) + 1).
        WaveTracker.frechet, csr_matrix      : 2D array of shape (nrays, nx*ny) in sparse csr format containing derivatives of travel
                                               time with respect to input velocity (velocityderiv=True) or slowness (velocityderiv=False) model values.                                               
    Notes:
        Internally variables are converted to numpy.float32 to be consistent with Fortran code fm2dss.f90

```
## Example

```python
import numpy as np
import pyfm2d as wt         # wavefront tracking package

m = np.array([[1,1.1,1.1,1.],
              [1.,1.2,1.4,1.3],
              [1.1,1.2,1.3,1.2],
              [1.1,1.1,1.2,1.2]])
g = wt.basisModel(m)
srcs = np.array([0.1,0.15])
recs = np.array([[0.8,1],[1.,0.6]])

fmm = wt.WaveTracker()
fmm.calc_wavefronts(g.getVelocity(),recs,srcs)

```
More detailed examples of its usage can be found in

[`examples/FMM_demo_borehole.ipynb`](./examples/FMM_demo_borehole.ipynb) - Cross borehole example

[`examples/FMM_demo Surface Wave.ipynb`](./examples/FMM_demo%20Surface%20Wave.ipynb) - Surface waves across Australia

[`examples/FMM_demo checkerboard.ipynb`](./examples/FMM_demo%20checkerboard.ipynb) - 2D checkerboard model.

## Gallery

A gallery of images produced by the plot class showing examples of raypaths and wavefront with severe ray bending can be found in directory [gallery](./gallery)

## Wrapping Strategy

The classes above call ctype wrapper functions that allow communication with the original Fortran code.
The idea is to refactor the original Fortran main into a subroutine inside a module that contains all
variables used by main as global variables. That is they are moved out of the subroutine.
As a consequence they have global scope and exist even when the main, that
is now a subroutine, is terminated. Thus they are accessible from Python via to be written
get and set functions.

## Tests

Running `test_fmmin2d.py` from within the test directory will run the orignal program
turned into a subroutine that can be called from python like the fmm executable it reads
the files from disk

Running `test_run.py` illustrates how the reorganised/expanded `fmm2dss.f90` now
allows in python to read the sources from disk by providing a file name, set them
from python and get them back from an fmm instance.

Running `test_wavetracker.py` illustrates how the wavetracker interface class utilizes the ctype interface to `fm2dss.f90`.

## Licensing
`pyfm2d` is released as BSD-2-Clause licence

## Citations and Acknowledgments

> *Rawlinson, N., de Kool, M. and Sambridge, M., 2006. "Seismic wavefront tracking in 3-D heterogeneous media: applications with multiple data classes", Explor. Geophys., 37, 322-330.*

> *de Kool, M., Rawlinson, N. and Sambridge, M. 2006. "A practical grid based method for tracking multiple refraction and reflection phases in 3D heterogeneous media", Geophys. J. Int., 167, 253-270.*
