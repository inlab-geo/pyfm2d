# pyfm2d

![Python3](https://img.shields.io/badge/python-3.x-brightgreen.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

_2D ray tracing and wavefront tracking_


This repository contains ctypes based Python wrappers for the program `fm2dss.F90` by Nick Rawlinson for 2D ray tracing and wavefront tracking using the Fast Marching method in a 2D seismic velocity or slowness model.


## Installation

PyFM2D can be directly installed from this repository
```
pip install git+https://github.com/inlab-geo/pyfm2d
```

Alternatively it is available on pypi and can be installed with

```
pip install pyfm2d
```

## Documentation

This is a package of several elements.

`BasisModel` - A class which contains functions to define, retrieve and manipulate 2D seismic velocity, or slowness, models expressed in a choice of model bases.

`WaveTrackerOptions` - A dataclass to allow adjustment of all parameters which control behaviour of the function `calc_wavefronts()`

`calc_wavefronts()` - the main function which calculates first arrival travel times between sources and receivers in a 2D model. Options for behavior are supplied by the options variable which is an instance of the class `WaveTrackerOptions`

`display_model()` - a function to plot velocity models together with calculated rays and wavefronts. 

Here is the docstring of the function `calc_wavefronts()` which does the work:

 	A function to perform 2D Fast Marching of wavefronts from sources in a 2D velocity model.

    Inputs:
        v, ndarray(nx,ny)          : coefficients of velocity field in 2D grid with dimension (nx,ny).
        recs, ndarray(nr,2)        : receiver locations (x,y). Where nr is the number of receivers.
        srcs, ndarray(ns,2)        : source locations (x,y). Where ns is the number of receivers.
        extent, list               : 4-tuple of model extent [xmin,xmax,ymin,ymax]. (default=[0.,1.,0.,1.])
        options, WaveTrackerOptions: configuration options for the wavefront tracker. (default=None)

    Returns
        WaveTrackerResult: a dataclass containing the results of the wavefront tracking.

    Notes:
        Internally variables are converted to np.float32 to be consistent with Fortran code fm2dss.f90.

## Example
A minimal example of its calling sequences is

```python
import numpy as np
import pyfm2d as wt         # wavefront tracking package

m = np.array([[1,1.1,1.1,1.],
              [1.,1.2,1.4,1.3],
              [1.1,1.2,1.3,1.2],
              [1.1,1.1,1.2,1.2]])
g = wt.BasisModel(m)
srcs = np.array([0.1,0.15])
recs = np.array([[0.8,1],[1.,0.6]])

wt.calc_wavefronts(g.get_velocity(),recs,srcs)

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
