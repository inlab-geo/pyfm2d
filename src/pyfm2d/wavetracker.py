import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csr_matrix, hstack
import faulthandler
from dataclasses import dataclass
from typing import Any, Callable, Optional
import concurrent.futures
from functools import reduce
import operator

from . import fastmarching as fmm
from . import bases as base


faulthandler.enable()

# --------------------------------------------------------------------------------------------

# This library is a python interface to Nick Rawlinson's 2D Fast Marching Fortran package fm2dss.f90
#
# It uses the ctypes interface to fm2dss.f90 developed by Juerg Hauser in file "._pyfm2dss"
#
# History:
#        January 2025:  Uses ctypes interface in pyfm2dss.py
#        January 2024:  Updated to interface with package bases.py allowing multiple 2D model bases
#                       including pixel and discrete cosine. Also interfaces with Jacobian integration
#                       package jip.py to calculate Frechect kernels when model bases
#                       are not simple pixel bases, e.g discrete cosine bases.
#
#        Definitions within waveTracker follow conventions from Andrew Valentine's rayTracer.py package.
#
# M. Sambridge
# January 2025
# --------------------------------------------------------------------------------------------


class Error(Exception):
    """Base class for other exceptions"""

    pass


class InputError(Exception):
    """Raised when necessary inputs are missing"""

    def __init__(self, msg=""):
        super().__init__(msg)


@dataclass
class WaveTrackerResult:
    """
    Result class for the calc_wavefronts function.

    Attributes:

        ttimes (np.ndarray): first arrival travel times between ns sources and nr receivers.

        paths (list): list of 2-D arrays (x,y) for ray paths between ns sources and nr receivers.

        ttfield (np.ndarray): 2-D array of travel time field for source tfieldsource at resolution mx*my

        frechet (csr_matrix): 2D array of shape (nrays, nx*ny) in sparse csr format containing derivatives of travel

    """

    ttimes: Optional[np.ndarray] = None
    paths: Optional[list] = None
    ttfield: Optional[np.ndarray] = None
    frechet: Optional[csr_matrix] = None

    def __add__(self, other: "WaveTrackerResult"):

        try:
            self._check_compatibility(other)
        except InputError as e:
            raise InputError(f"Incompatible WaveTrackerResults: {e}")

        if self.ttimes is not None:
            ttimes = np.concatenate([self.ttimes, other.ttimes])
        else:
            ttimes = None
        if self.paths is not None:
            paths = self.paths + other.paths
        else:
            paths = None
        if self.ttfield is not None:
            ttfield = np.concatenate([self.ttfield, other.ttfield])
        else:
            ttfield = None
        if self.frechet is not None:
            frechet = hstack([self.frechet, other.frechet])
        else:
            frechet = None

        return WaveTrackerResult(ttimes, paths, ttfield, frechet)

    def _check_compatibility(self, other: "WaveTrackerResult"):
        if (self.ttimes is None) != (other.ttimes is None):
            raise InputError("Travel times are not available for both results.")
        if (self.paths is None) != (other.paths is None):
            raise InputError("Ray paths are not available for both results.")
        if (self.ttfield is None) != (other.ttfield is None):
            raise InputError("Travel time fields are not available for both results.")
        if (self.frechet is None) != (other.frechet is None):
            raise InputError("Frechet derivatives are not available for both results.")


@dataclass
class WaveTrackerOptions:
    """
    WaveTrackerOptions is a configuration class for the calc_wavefronts function.

    Attributes:

        times (bool): Whether to calculate travel times. Default is True.

        paths (bool): Whether to calculate ray paths. Default is False.

        frechet (bool): Whether to compute Frechet derivatives. Default is False.

        ttfield_source (int): Source index for to compute travel time field. If <0 then no fields are computed. Default is -1.

        sourcegridrefine (bool): Apply sourcegrid refinement. Default is True.

        sourcedicelevel (int): Source discretization level. Number of sub-divisions per cell (default=5, i.e. 1 model cell becomes 5x5 sub-cells)

        sourcegridsize (int): Number of model cells to refine about source at sourcedicelevel (default=10, i.e. 10x10 cells are refines about source)

        earthradius (float): radius of Earth in km, used for spherical to Cartesian transform. Default is 6371.0.

        schemeorder (int): switch to use first order (0) or mixed order (1) scheme. Default is 1.

        nbsize (float): Narrow band size (0-1) as fraction of nnx*nnz. Default is 0.5.

        degrees (bool): True if input distances are in degrees. Default is False.

        velocityderiv (bool): Switch to return Frechet derivatives of travel times w.r.t. velocities (True) rather than slownesses (False). Default is False.

        dicex (int): x-subgrid discretization level for B-spline interpolation of input mode. Default is 8.

        dicey (int): y-subgrid discretization level for B-spline interpolation of input model. Default is 8.
    """

    times: bool = True
    paths: bool = False
    frechet: bool = False
    ttfield_source: int = -1
    sourcegridrefine: bool = True
    sourcedicelevel: int = 5
    sourcegridsize: int = 10
    earthradius: float = 6371.0
    schemeorder: int = 1
    nbsize: float = 0.5
    degrees: bool = False
    velocityderiv: bool = False
    dicex: int = 8
    dicey: int = 8

    def __post_init__(self):
        # mostly convert boolean to int for Fortran compatibility

        # Write out ray paths. Only allow all (-1) or none (0)
        self.lpaths: int = -1 if self.paths else 0

        # int to calculate travel times (y=1,n=0)
        self.lttimes: int = 1 if self.times else 0

        # int to calculate Frechet derivatives of travel times w.r.t. slownesses (0=no,1=yes)
        self.lfrechet: int = 1 if self.frechet else 0

        # int to calculate travel fields (0=no,1=all)
        self.tsource: int = 1 if self.ttfield_source >= 0 else 0


def cleanup(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            fmm.deallocate_result_arrays()
            raise e

    return wrapper


@cleanup
def calc_wavefronts(
    v,
    recs,
    srcs,
    extent=[0.0, 1.0, 0.0, 1.0],
    options: Optional[WaveTrackerOptions] = None,
    nthreads: int = 1,
):
    """

    A function to perform 2D Fast Marching of wavefronts from sources in a 2D velocity model.

    Inputs:
        v, ndarray(nx,ny)          : coefficients of velocity field in 2D grid with dimension (nx,ny).
        recs, ndarray(nr,2)        : receiver locations (x,y). Where nr is the number of receivers.
        srcs, ndarray(ns,2)        : source locations (x,y). Where ns is the number of receivers.
        extent, list               : 4-tuple of model extent [xmin,xmax,ymin,ymax]. (default=[0.,1.,0.,1.])
        options, WaveTrackerOptions: configuration options for the wavefront tracker. (default=None)
        nthreads, int              : number of threads to use for multithreading. Multithreading is performed over sources (default=1)


    Returns
        WaveTrackerResult: a dataclass containing the results of the wavefront tracking.

    Notes:
        Internally variables are converted to np.float32 to be consistent with Fortran code fm2dss.f90.

    """

    if nthreads <= 1:
        return _calc_wavefronts_process(v, recs, srcs, extent, options)
    else:
        return _calc_wavefronts_multithreading(v, recs, srcs, nthreads, extent, options)


def _calc_wavefronts_process(
    v,
    recs,
    srcs,
    extent=[0.0, 1.0, 0.0, 1.0],
    options: Optional[WaveTrackerOptions] = None,
):
    # here extent[3],extent[2] is N-S range of grid nodes
    #      extent[0],extent[1] is W-E range of grid nodes
    if options is None:
        options = WaveTrackerOptions()

    recs = recs.reshape(-1, 2)  # ensure receiver array is 2D and float32
    srcs = srcs.reshape(-1, 2)  # ensure source array is 2D and float32

    _check_sources_receivers_inside_extent(srcs, recs, extent)

    _check_requested_source_exists(options.ttfield_source, len(srcs))

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
    dlat = (extent[3] - extent[2]) / (nvy - 1)
    dlong = (extent[1] - extent[0]) / (nvx - 1)

    vc = _build_velocity_grid(v)

    fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)

    # set up time calculation between all sources and receivers
    associations = np.ones((recs.shape[0], srcs.shape[0]))
    fmm.set_source_receiver_associations(associations)

    fmm.allocate_result_arrays()  # allocate memory for Fortran arrays

    fmm.track()  # run fmst wavefront tracker code

    # collect results
    result = collect_results(options, v)

    fmm.deallocate_result_arrays()

    return result


def _calc_wavefronts_multithreading(
    v,
    recs,
    srcs,
    nthreads=2,
    extent=[0.0, 1.0, 0.0, 1.0],
    options: Optional[WaveTrackerOptions] = None,
) -> WaveTrackerResult:

    # Since this function is called when there are multiple sources, we can't specify a single source for the full field calcutlation
    # Although we could create a list of source indices...
    options.ttfield_source = -1

    futures = []
    # https://docs.python.org/3/library/concurrent.futures.html
    with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
        for i in range(np.shape(srcs)[0]):
            futures.append(
                executor.submit(
                    _calc_wavefronts_process, v, recs, srcs[i, :], extent, options
                )
            )
    result_list = [f.result() for f in futures]

    return reduce(operator.add, result_list)


def collect_results(options: WaveTrackerOptions, velocity):
    # fmst expects input spatial co-ordinates in degrees and velocities in kms/s so we adjust (unless degrees=True)
    kms2deg = 1.0 if options.degrees else 180.0 / (options.earthradius * np.pi)

    ttimes = None
    raypaths = None
    frechetvals = None
    tfield = None

    if options.times:
        ttimes = fmm.get_traveltimes().copy() * kms2deg

    if options.paths:
        raypaths = fmm.get_raypaths().copy()

    if options.frechet:
        frechetvals = _get_frechet_derivatives(kms2deg, options.velocityderiv, velocity)

    if options.ttfield_source >= 0:
        tfield = _get_tfield(kms2deg, options.ttfield_source)

    return WaveTrackerResult(ttimes, raypaths, tfield, frechetvals)


def _get_frechet_derivatives(degrees_conversion, velocityderiv, velocity):
    frechetvals = fmm.get_frechet_derivatives()
    frechetvals *= degrees_conversion

    # the frechet matrix returned in in csr format and has two layers of cushion nodes surrounding the (nx,ny) grid
    F = frechetvals.toarray()  # unpack csr format
    nrays = F.shape[0]  # number of raypaths
    nx, ny = velocity.shape  # shape of non-cushion velcoity model

    # remove cushion nodes and reshape to (nx,ny)
    noncushion = _build_grid_noncushion_map(nx, ny)
    F = F[:, noncushion.flatten()].reshape((nrays, nx, ny))

    # reverse y order, because it seems to be returned in reverse order (cf. ttfield array)
    F = F[:, :, ::-1]

    # reformat as a sparse CSR matrix
    frechetvals = csr_matrix(F.reshape((nrays, nx * ny)))

    if not velocityderiv:
        # adjust derivatives to be of velocites rather than slownesses (default)
        x2 = -(velocity * velocity).reshape(-1)
        frechetvals = frechetvals.multiply(x2)

    return frechetvals.copy()


def _get_tfield(degrees_conversion, source):
    tfieldvals = fmm.get_traveltime_fields()
    tfieldvals *= degrees_conversion

    tfield = tfieldvals[source].copy()

    # flip y axis of travel time field as it is provided in reverse ordered.
    tfield = tfield[:, ::-1]
    return tfield


def _build_velocity_grid(v):  # maybe this could be split up a bit?
    # add cushion nodes about velocity model to be compatible with fm2dss.f90 input

    nx, ny = v.shape

    # gridc.vtx requires a single cushion layer of nodes surrounding the velocty model
    # additional boundary layer of velocities are duplicates of the nearest actual velocity value.
    vc = np.ones((nx + 2, ny + 2))
    vc[1 : nx + 1, 1 : ny + 1] = v
    vc[1 : nx + 1, 0] = v[:, 0]  # add velocities in the cushion boundary layer
    vc[1 : nx + 1, -1] = v[:, -1]  # add velocities in the cushion boundary layer
    vc[0, 1 : ny + 1] = v[0, :]  # add velocities in the cushion boundary layer
    vc[-1, 1 : ny + 1] = v[-1, :]  # add velocities in the cushion boundary layer
    vc[0, 0], vc[0, -1], vc[-1, 0], vc[-1, -1] = (
        v[0, 0],
        v[0, -1],
        v[-1, 0],
        v[-1, -1],
    )
    vc = vc[:, ::-1]

    return vc


def _build_grid_noncushion_map(nx, ny):
    # bool array to identify cushion and non cushion nodes
    noncushion = np.zeros((nx + 2, ny + 2), dtype=bool)
    noncushion[1 : nx + 1, 1 : ny + 1] = True
    return noncushion


def _build_node_map(nx, ny):
    # mapping from cushion indices to non cushion indices
    nodemap = np.zeros((nx + 2, ny + 2), dtype=int)
    nodemap[1 : nx + 1, 1 : ny + 1] = np.array(range((nx * ny))).reshape((nx, ny))
    nodemap = nodemap[:, ::-1]
    return nodemap


def _check_sources_receivers_inside_extent(srcs, recs, extent):
    xmin = extent[0]
    xmax = extent[1]
    ymin = extent[2]
    ymax = extent[3]

    rcx = recs[:, 0]
    rcy = recs[:, 1]
    if not np.all((xmin <= rcx) & (rcx <= xmax) & (ymin <= rcy) & (rcy <= ymax)):
        raise InputError(
            msg="Input Error: One or more receiver lies outside of model extent: "
            + str(extent)
            + "\nRemedy: adjust receiver locations and run again."
        )

    srcx = srcs[:, 0]
    srcy = srcs[:, 1]
    if not np.all((xmin <= srcx) & (srcx <= xmax) & (ymin <= srcy) & (srcy <= ymax)):
        raise InputError(
            msg="Input Error: One or more source lies outside of model extent: "
            + str(extent)
            + "\nRemedy: adjust source locations and run again."
        )


def _check_requested_source_exists(tfieldsource, ns):
    if tfieldsource + 1 > ns:
        # source requested for travel time field does not exist
        print(
            f"Error: Travel time field corresponds to source: {tfieldsource}",
            "\n",
            f"      but total number of sources is {ns}.",
            "\n       No travel time field will be calculated.\n",
        )


class GridModel:  # This is for the original regular model grid (without using the basis.py package)

    def __init__(self, velocities, extent=(0, 1, 0, 1)):
        self.nx, self.ny = velocities.shape
        self.velocities = velocities
        self.xmin, self.xmax, self.ymin, self.ymax = extent
        self.xx = np.linspace(self.xmin, self.xmax, self.nx + 1)
        self.yy = np.linspace(self.ymin, self.ymax, self.ny + 1)
        self.extent = extent

    def get_velocity(self):
        return self.velocities.copy()

    def get_slowness(self):
        return 1.0 / self.velocities  # No copy needed as operation must return copy

    def set_velocity(self, v):
        assert self.velocities.shape == v.shape
        self.velocities = v.copy()

    def set_slowness(self, s):
        assert self.velocities.shape == s.shape
        self.velocities = 1.0 / s


class BasisModel:  # This is for a 2D model basis accessed through the package basis.py
    """

    A model class which is an interface to package basis.py to incorporate local or global 2D bases for tomography.

    Handles cases where model bases are local 2D pixels (default) or global 2D functions, e.g. cosine basis.
    Handles cases where input coefficients are velocities (default) or slownesses.

    Inputs:
        coeffs, ndarray(nx,ny)          : coefficients of velocity or slowness field in selected basis
        coeff_type, string              : ='velocities' then coefficients are velocities; 'slownesses' for slowness coefficients
        basis, string                   : type of model basis function
                                          `2Dpixel' for 2D regular grid of velocity/slowness values;
                                          '2Dcosine' for 2D cosine basis functions.
        ref, float                      : reference value used for perturbative representation,
                                          i.e. v[x,y] = ref + coeff[i]*basis[i,x,y]; or s[x,y] = ref + coeff[i]*basis[i,x,y], (i=1,...nx*ny)

    """

    def __init__(
        self,
        coeffs,
        extent=(0, 1, 0, 1),
        ref=0.0,
        coeff_type="velocities",
        basis="2Dpixel",
    ):
        self.nx, self.ny = coeffs.shape
        self.coeffs = coeffs
        self.xmin, self.xmax, self.ymin, self.ymax = extent
        self.extent = extent
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        self.basis_type = basis
        self.A_calc = False
        self.vref, self.sref = 0.0, 0.0  # reference values
        if ref != 0.0:
            if coeff_type == "velocities":
                self.vref, self.sref = ref, 1.0 / ref
            else:
                self.vref, self.sref = 1.0 / ref, ref
        if self.basis_type == "2Dpixel":
            self.xx = np.linspace(self.xmin, self.xmax, self.nx + 1)
            self.yy = np.linspace(self.ymin, self.ymax, self.ny + 1)
            self.basis = base.PixelBasis2D(self.xx, self.yy)
        elif self.basis_type == "2Dcosine":
            self.basis = base.CosineBasis2D(
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
                self.nx,
                self.ny,
                npoints=[120, 120, 200],
            )

        self.coeff_type = coeff_type  # need to know this for non-pixel bases

    def get_velocity(self, nx=None, ny=None, returncoeff=False):
        """With no arguments this will return a velocity field.
        If bases are 2D pixels the keyword returncoeff is ignored
        If bases are not pixels and returncoeff is False, then a velocity field is evaluated and returned.
        If bases are not 2D pixels and returncoeff is True, then velocity basis coefficients are returned.
        Default values of (nx,ny) are determined by .getImage() and are the input resolution of the model.
        """
        if self.basis_type == "2Dpixel":
            if self.coeff_type == "velocities":
                if returncoeff:
                    return self.coeffs.copy()
                return self.vref + self.coeffs.copy()
            else:  # slownesses
                if returncoeff:
                    return 1.0 / self.coeffs
                return 1.0 / (self.sref + self.coeffs)
        else:  # non pixel basis
            if self.coeff_type == "velocities":
                if returncoeff:
                    return self.coeffs.copy()
                else:
                    # return a velocity field evaluated from basis summation
                    return self.vref + self.get_image(nx=nx, ny=ny)
            else:  # slownesses
                if returncoeff:
                    # coefficients are slownesses and we need to find equivalent velocity coefficients
                    return self.fit_coefficients_s2v()
                else:
                    # coefficients are slownesses and we must return a velocity field
                    return 1.0 / (self.sref + self.get_image(nx=nx, ny=ny))

    def basis_transform_matrix(self):
        if not self.A_calc:
            A = np.zeros((self.basis.nbases, self.nx * self.ny))
            for j in range(self.basis.nbases):
                A[j] = self.get_basis_image(j, nx=self.nx, ny=self.ny).flatten()
            self.A = A
            self.A_calc = True
        return self.A

    def fit_coefficientes_v2s(self, nx=None, ny=None):
        """
        calculate slowness coefficients that correspond to a given set of velocity coefficients in model basis
        """
        if nx is None:
            nx = self.nx
        if ny is None:
            ny = self.ny
        vtarget = self.get_velocity(nx=nx, ny=ny) - self.vref
        if not self.A_calc:
            A = self.basis_transform_matrix()
            self.A = A
        slowcoeff, res, rank, s = np.linalg.lstsq(
            self.A.T, 1.0 / vtarget.flatten(), rcond=None
        )  # fit slownesses coefficients to slowness field
        return slowcoeff.reshape(self.nx, self.ny)

    def fit_coefficients_s2v(self, nx=None, ny=None):
        """
        calculate velocity coefficients that correspond to a given set of slowness coefficients in model basis
        """
        if nx is None:
            nx = self.nx
        if ny is None:
            ny = self.ny
        starget = (
            self.get_slowness(nx=nx, ny=ny) - self.sref
        )  # get slowness field perturbation
        if not self.A_calc:
            A = self.basis_transform_matrix()
            self.A = A
        velcoeff, res, rank, s = np.linalg.lstsq(
            self.A.T, 1.0 / starget.flatten(), rcond=None
        )  # fit slownesses coefficients to slowness field
        return velcoeff.reshape(self.nx, self.ny)

    def convert_pixel_vel_2_basis_slow(self, v):
        """
        convert velocity model in pixel basis to equivalent model as slowness coefficients in model basis
        """
        nx, ny = v.shape
        vpert = v - self.vref
        if np.all(vpert) == 0.0:
            return np.zeros_like(v)
        # coeff = self.coeffs.copy()
        if not self.A_calc:
            A = self.basis_transform_matrix()
            self.A = A
        slowcoeff, res, rank, s = np.linalg.lstsq(
            self.A.T, 1.0 / vpert.flatten(), rcond=None
        )  # fit slownesses coefficients to slowness field
        # self.setCoeffs(coeff)
        return slowcoeff.reshape(self.nx, self.ny)

    def convert_pixel_vel_2_basis_vel(self, v):
        """
        convert velocity model in pixel basis to equivalent model as velocity coefficients in model basis
        """
        nx, ny = v.shape
        vpert = v - self.vref
        # coeff = self.coeffs.copy()
        if not self.A_calc:
            A = self.basis_transform_matrix()
            self.A = A
        velcoeff, res, rank, s = np.linalg.lstsq(
            self.A.T, vpert.flatten(), rcond=None
        )  # fit slownesses coefficients to slowness field
        # self.setCoeffs(coeff)
        return velcoeff.reshape(self.nx, self.ny)

    def get_coeffs(self):
        return self.coeffs.copy()

    def set_coeffs(self, c):
        assert self.coeffs.shape == c.shape
        self.coeffs = c

    def get_slowness(self, nx=None, ny=None, returncoeff=False):
        """
        With no arguments this will return a slowness field.
        If bases are 2D pixels the keyword returncoeff is ignored
        If bases are not pixels and returncoeff is False, then a slowness field is evaluated and returned at (nx,ny)
        If bases are not 2D pixels and returncoeff is True, then slowness basis coefficients are returned.
        Default values of (nx,ny) are determined by .getImage() and are the input resolution of the model.
        """
        if self.basis_type == "2Dpixel":
            if self.coeff_type == "velocities":
                if returncoeff:
                    return 1.0 / (self.vref + self.coeffs) - self.sref
                return 1.0 / (self.vref + self.coeffs)
            else:  # slownesses
                if returncoeff:
                    return self.coeffs.copy()
                return self.sref + self.coeffs.copy()
        else:
            if self.coeff_type == "velocities":
                if returncoeff:
                    # we need to find slowness coefficients from velocity coefficients here
                    return self.fit_coefficientes_v2s()
                else:
                    # return a slowness field after summation of velocity bases
                    return 1.0 / (self.vref + self.get_image(nx=nx, ny=ny))
            else:
                if returncoeff:
                    # coefficients are slownesses and so we return coefficients
                    return self.coeffs.copy()
                else:
                    # return a slowness field after summation of slowness bases
                    return self.sref + self.get_image(nx=nx, ny=ny)

    def set_velocity(self, v):  # set Velocity coefficients
        if self.coeff_type == "velocities":
            assert self.coeffs.shape == v.shape
            self.coeffs = v.copy()
        else:
            if self.basis_type == "2Dpixel":
                assert self.coeffs.shape == v.shape
                self.coeffs = 1.0 / (v + self.vref) - self.sref
            else:
                print(
                    " Error: can not set velocity coefficients if coeff type are slownesses and basis is not pixel"
                )
                pass  # coefficients are slownesses and we need to find and set equivalent velocity coefficients (not implemented)

    def set_slowness(self, s):  # set Slowness coefficients
        if self.coeff_type == "velocities":
            if self.basis_type == "2Dpixel":
                assert self.coeffs.shape == s.shape
                self.coeffs = 1.0 / (s + self.sref) - self.vref
            else:
                print(
                    " Error: can not set slowness coefficients if coeff type are velocities and basis is not pixel"
                )
                pass  # coefficients are velocities and we need to find and set equivalent slowness coefficients (not implemented)
        else:
            assert self.coeffs.shape == s.shape
            self.coeffs = (
                s.copy()
            )  # coefficients are slownesses and so we set new slowness coefficients

    def get_image(
        self, nx=None, ny=None
    ):  # returns 2D image of model at chosen resolution
        if nx is None:
            nx = self.nx
        if ny is None:
            ny = self.ny
        dx, dy = (self.xmax - self.xmin) / nx, (self.ymax - self.ymin) / ny
        Ym, Xm = np.meshgrid(
            np.linspace(self.ymin + dy / 2, self.ymax - dy / 2, ny),
            np.linspace(self.xmin + dx / 2.0, self.xmax - dx / 2, nx),
        )
        image = np.zeros_like(Xm)
        for j in range(
            self.basis.nbases
        ):  # sum over bases and evaluate model at each pixel in image
            image += self.coeffs.flatten()[j] * (self.basis.evaluate(j, (Xm, Ym)))
        return image

    def get_basis_image(
        self, j, nx=None, ny=None
    ):  # returns 2D image of model at chosen resolution
        if nx is None:
            nx = self.nx
        if ny is None:
            ny = self.ny
        dx, dy = (self.xmax - self.xmin) / nx, (self.ymax - self.ymin) / ny
        Ym, Xm = np.meshgrid(
            np.linspace(self.ymin + dy / 2, self.ymax - dy / 2, ny),
            np.linspace(self.xmin + dx / 2.0, self.xmax - dx / 2, nx),
        )
        return self.basis.evaluate(j, (Xm, Ym))


# --------------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------------
def norm(x):
    return np.sqrt(x.dot(x))


def normalise(x):
    return x / norm(x)


def png_to_model(pngfile, nx, ny, bg=1.0, sc=1.0):
    png = Image.open(pngfile)
    png.load()
    model = sc * (
        bg
        + np.asarray(png.convert("L").resize((nx, ny)).transpose(Image.ROTATE_270))
        / 255.0
    )
    return model


def generate_surface_points(
    nPerSide,
    extent=(0, 1, 0, 1),
    surface=[True, True, True, True],
    addCorners=True,
):
    out = []
    x = np.linspace(extent[0], extent[1], nPerSide + 2)[1 : nPerSide + 1]
    y = np.linspace(extent[2], extent[3], nPerSide + 2)[1 : nPerSide + 1]
    if surface[0]:
        out += [[extent[0], _y] for _y in y]
    if surface[1]:
        out += [[extent[1], _y] for _y in y]
    if surface[2]:
        out += [[_x, extent[2]] for _x in x]
    if surface[3]:
        out += [[_x, extent[3]] for _x in x]
    if addCorners:
        if surface[0] or surface[2]:
            out += [[extent[0], extent[2]]]
        if surface[0] or surface[3]:
            out += [[extent[0], extent[3]]]
        if surface[1] or surface[2]:
            out += [[extent[1], extent[2]]]
        if surface[1] or surface[3]:
            out += [[extent[1], extent[3]]]
    return np.array(out)


# --------------------------------------------------------------------------------------------
# Plotting routines
# --------------------------------------------------------------------------------------------


def display_model(
    model,
    paths=None,
    extent=(0, 1, 0, 1),
    clim=None,
    cmap=None,
    figsize=(6, 6),
    title=None,
    line=1.0,
    cline="k",
    alpha=1.0,
    wfront=None,
    cwfront="k",
    diced=True,
    dicex=8,
    dicey=8,
    cbarshrink=0.6,
    cbar=True,
    filename=None,
    **wkwargs,
):
    """

    Function to plot 2D velocity or slowness field

    Inputs:
        model, ndarray(nx,ny)           : 2D velocity or slowness field on rectangular grid
        paths, string                   :

    """

    plt.figure(figsize=figsize)
    if cmap is None:
        cmap = plt.cm.RdBu

    # if diced option plot the actual B-spline interpolated velocity used by fmst program

    plotmodel = model
    if diced:
        plotmodel = create_diced_grid(model, extent=extent, dicex=dicex, dicey=dicey)

    plt.imshow(plotmodel.T, origin="lower", extent=extent, cmap=cmap)

    if paths is not None:
        if isinstance(paths, np.ndarray) and paths.ndim == 2:
            if paths.shape[1] == 4:  # we have paths from xrt.tracer so adjust
                paths = change_paths_format(paths)

        for p in paths:
            plt.plot(p[:, 0], p[:, 1], cline, lw=line, alpha=alpha)

    if clim is not None:
        plt.clim(clim)

    if title is not None:
        plt.title(title)

    if wfront is not None:
        nx, ny = wfront.shape
        X, Y = np.meshgrid(
            np.linspace(extent[0], extent[1], nx),
            np.linspace(extent[2], extent[3], ny),
        )
        plt.contour(X, Y, wfront.T, **wkwargs)  # Negative contours default to dashed.

    if wfront is None and cbar:
        plt.colorbar(shrink=cbarshrink)

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def create_diced_grid(v, extent=[0.0, 1.0, 0.0, 1.0], dicex=8, dicey=8):
    nx, ny = v.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    kx, ky = 3, 3
    if nx <= 3:
        kx = nx - 1  # reduce order of B-spline if we have too few velocity nodes
    if ny <= 3:
        ky = ny - 1
    rect = RectBivariateSpline(x, y, v, kx=kx, ky=ky)
    xx = np.linspace(extent[0], extent[1], dicex * nx)
    yy = np.linspace(extent[2], extent[3], dicey * ny)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    vinterp = rect.ev(X, Y)
    return vinterp


def change_paths_format(paths):
    p = np.zeros((len(paths), 2, 2))
    for i in range(len(paths)):
        p[i, 0, 0] = paths[i, 0]
        p[i, 0, 1] = paths[i, 1]
        p[i, 1, 0] = paths[i, 2]
        p[i, 1, 1] = paths[i, 3]
    return p
