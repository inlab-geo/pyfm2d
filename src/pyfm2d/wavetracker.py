import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csr_matrix, vstack
import faulthandler
from dataclasses import dataclass
from typing import Optional
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

        ttimes = np.concatenate([self.ttimes, other.ttimes]) if self.ttimes is not None else None
        paths = self.paths + other.paths if self.paths is not None else None
        ttfield = np.concatenate([self.ttfield, other.ttfield]) if self.ttfield is not None else None
        frechet = vstack([self.frechet, other.frechet]) if self.frechet is not None else None

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

        ttfield_source (int): Source index for computation of travel time field. If <0 then no fields are computed. Default is -1.

        sourcegridrefine (bool): Apply sourcegrid refinement. Default is True.

        sourcedicelevel (int): Source discretization level. Number of sub-divisions per cell (default=5, i.e. 1 model cell becomes 5x5 sub-cells)

        sourcegridsize (int): Number of model cells to refine about source at sourcedicelevel (default=10, i.e. 10x10 cells are refined about source)

        earthradius (float): radius of Earth in km, used for spherical to Cartesian transform. Default is 6371.0.

        schemeorder (int): switch to use first order (0) or mixed order (1) scheme. Default is 1.

        nbsize (float): Narrow band size (0-1) as fraction of nnx*nnz. Default is 0.5.

        cartesian (bool): True if using a Cartesian spatial frame. Default is False.

        velocityderiv (bool): Switch to return Frechet derivatives of travel times w.r.t. velocities (True) rather than slownesses (False). Default is False.

        dicex (int): x-subgrid discretization level for B-spline interpolation of input mode. Default is 8.

        dicey (int): y-subgrid discretization level for B-spline interpolation of input model. Default is 8.

        quiet (bool): Suppress non-fatal ray path and boundary warnings. Default is False (show warnings).
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
    cartesian: bool = False
    velocityderiv: bool = False
    dicex: int = 8
    dicey: int = 8
    quiet: bool = False

    def __post_init__(self):
        # mostly convert boolean to int for Fortran compatibility

        # Frechet derivatives require paths to be computed (npaths counter is used for row indices)
        if self.frechet and not self.paths:
            import warnings
            warnings.warn(
                "frechet=True requires paths=True for correct row indexing. "
                "Setting paths=True automatically.",
                UserWarning
            )
            object.__setattr__(self, 'paths', True)

        # Write out ray paths. Only allow all (-1) or none (0)
        self.lpaths: int = -1 if self.paths else 0

        # int to calculate travel times (y=1,n=0)
        self.lttimes: int = 1 if self.times else 0

        # int to calculate Frechet derivatives of travel times w.r.t. slownesses (0=no,1=yes)
        self.lfrechet: int = 1 if self.frechet else 0

        # int to calculate travel fields (0=no,1=all)
        self.tsource: int = 1 if self.ttfield_source >= 0 else 0

        # int to activate cartesian mode (y=1,n=0)
        self.lcartesian: int = 1 if self.cartesian else 0

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
    pool=None,
    quiet: Optional[bool] = None,
    associations: Optional[np.ndarray] = None,
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
        pool                       : User-provided pool for parallel processing. If provided, this takes precedence
                                    over the nthreads parameter. The pool must implement either a submit() method
                                    (like concurrent.futures executors) or a map() method (like schwimmbad pools).
                                    When providing a pool, the user is responsible for its lifecycle management.
                                    (default=None)
        quiet, bool                : Suppress non-fatal ray path and boundary warnings. If provided, overrides
                                    options.quiet. (default=None)
        associations, ndarray(nr,ns): Binary matrix indicating which source-receiver pairs to compute.
                                    Shape is (n_receivers, n_sources). A value of 1 at [i,j] means compute
                                    the travel time from source j to receiver i. If None, computes all pairs.
                                    (default=None)


    Returns
        WaveTrackerResult: a dataclass containing the results of the wavefront tracking.

    Notes:
        Internally variables are converted to np.float32 to be consistent with Fortran code fm2dss.f90.

    """

    # Initialize options if not provided
    if options is None:
        options = WaveTrackerOptions()

    # Direct parameter overrides options.quiet if provided
    if quiet is not None:
        import dataclasses
        options = dataclasses.replace(options, quiet=quiet)

    if pool is not None:
        # Check if pool is a ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        if isinstance(pool, ThreadPoolExecutor):
            raise ValueError(
                "ThreadPoolExecutor is not supported due to shared memory conflicts in the "
                "underlying Fortran implementation. Multiple threads cannot safely allocate "
                "the same global Fortran arrays. Please use ProcessPoolExecutor or other "
                "process-based pools instead."
            )
        # User-provided pool takes precedence
        return _calc_wavefronts_multithreading(v, recs, srcs, extent, options, pool=pool, associations=associations)
    elif nthreads <= 1:
        return _calc_wavefronts_process(v, recs, srcs, extent, options, associations=associations)
    else:
        return _calc_wavefronts_multithreading(v, recs, srcs, extent, options, nthreads=nthreads, associations=associations)


def _calc_wavefronts_process(
    v,
    recs,
    srcs,
    extent=[0.0, 1.0, 0.0, 1.0],
    options: Optional[WaveTrackerOptions] = None,
    associations: Optional[np.ndarray] = None,
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
        options.lcartesian,
        int(options.quiet),
    )

    fmm.set_sources(srcs[:, 1], srcs[:, 0])  # ordering inherited from fm2dss.f90
    fmm.set_receivers(recs[:, 1], recs[:, 0])  # ordering inherited from fm2dss.f90

    nvx, nvy = v.shape
    # grid node spacing in lat and long
    dlat = (extent[3] - extent[2]) / (nvy - 1)
    dlong = (extent[1] - extent[0]) / (nvx - 1)

    vc = _build_velocity_grid(v)

    if (options.lcartesian == 1): # grid in regular order if Cartesian mode (co-ords are in kms)
        fmm.set_velocity_model(nvy, nvx, extent[2], extent[0], dlat, dlong, vc)

    else:                         # y-grid (Lat) required in reversed order if Spherical mode (co-ords are in degs)

        vc = vc[:, ::-1] # reverse direction of velocity model in latitude direction for Spherical model
        fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)

    # set up time calculation between sources and receivers
    if associations is None:
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
    extent=[0.0, 1.0, 0.0, 1.0],
    options: Optional[WaveTrackerOptions] = None,
    nthreads=2,
    pool=None,
    associations: Optional[np.ndarray] = None,
) -> WaveTrackerResult:

    # Since this function is called when there are multiple sources, we can't specify a single source for the full field calcutlation
    # Although we could create a list of source indices...
    options.ttfield_source = -1

    # Check if user provided a pool
    if pool is not None:
        created_pool = None
        executor = pool
    else:
        # Create internal ProcessPoolExecutor
        created_pool = concurrent.futures.ProcessPoolExecutor(max_workers=nthreads)
        executor = created_pool

    try:
        # Check if executor has submit method (concurrent.futures style)
        if hasattr(executor, 'submit'):
            futures = []
            for i in range(np.shape(srcs)[0]):
                # Extract associations column for this source (reshape to 2D for single source)
                src_associations = associations[:, i:i+1] if associations is not None else None
                futures.append(
                    executor.submit(
                        _calc_wavefronts_process, v, recs, srcs[i, :], extent, options, src_associations
                    )
                )
            result_list = [f.result() for f in futures]
        else:
            # Use map for pools that don't have submit (e.g., schwimmbad pools)
            # Create a list of arguments for each source
            args_list = [
                (v, recs, srcs[i, :], extent, options, associations[:, i:i+1] if associations is not None else None)
                for i in range(np.shape(srcs)[0])
            ]
            result_list = list(executor.map(lambda args: _calc_wavefronts_process(*args), args_list))
    finally:
        # Clean up internally created pool
        if created_pool is not None:
            created_pool.shutdown(wait=True)

    return reduce(operator.add, result_list)


def collect_results(options: WaveTrackerOptions, velocity):
    """Collect results from the Fortran solver based on requested options."""
    ttimes = fmm.get_traveltimes().copy() if options.times else None
    raypaths = fmm.get_raypaths().copy() if options.paths else None
    frechetvals = _get_frechet_derivatives(options.cartesian, options.velocityderiv, velocity) if options.frechet else None
    tfield = _get_tfield(options.cartesian, options.ttfield_source) if options.ttfield_source >= 0 else None

    return WaveTrackerResult(ttimes, raypaths, tfield, frechetvals)


def _get_frechet_derivatives(cartesian, velocityderiv, velocity):
    """Extract and process Frechet derivatives from the Fortran solver."""
    frechetvals = fmm.get_frechet_derivatives()
    nx, ny = velocity.shape

    # Remove cushion nodes and reshape
    F = frechetvals.toarray()
    nrays = F.shape[0]
    noncushion = _build_grid_noncushion_map(nx, ny)
    F = F[:, noncushion.flatten()].reshape((nrays, nx, ny))

    # For Spherical mode: reverse y order for consistency with ttfield array
    if not cartesian:
        F = F[:, :, ::-1]

    frechetvals = csr_matrix(F.reshape((nrays, nx * ny)))

    # Convert to slowness derivatives (default) unless velocity derivatives requested
    if not velocityderiv:
        x2 = -(velocity * velocity).reshape(-1)
        frechetvals = frechetvals.multiply(x2)

    return frechetvals.copy()


def _get_tfield(cartesian, source):
    """Extract travel time field for a specific source."""
    tfield = fmm.get_traveltime_fields()[source].copy()
    # Flip y axis for spherical mode (field is provided in reverse order)
    if not cartesian:
        tfield = tfield[:, ::-1]
    return tfield


def _build_velocity_grid(v):
    """Add cushion nodes around velocity model for fm2dss.f90 compatibility."""
    nx, ny = v.shape
    vc = np.ones((nx + 2, ny + 2))

    # Copy interior values
    vc[1:nx + 1, 1:ny + 1] = v

    # Extend boundary values to cushion layer
    vc[1:nx + 1, 0] = v[:, 0]
    vc[1:nx + 1, -1] = v[:, -1]
    vc[0, 1:ny + 1] = v[0, :]
    vc[-1, 1:ny + 1] = v[-1, :]

    # Corner values
    vc[0, 0], vc[0, -1], vc[-1, 0], vc[-1, -1] = v[0, 0], v[0, -1], v[-1, 0], v[-1, -1]

    return vc


def _build_grid_noncushion_map(nx, ny):
    """Create boolean array identifying non-cushion nodes."""
    noncushion = np.zeros((nx + 2, ny + 2), dtype=bool)
    noncushion[1:nx + 1, 1:ny + 1] = True
    return noncushion


def _build_node_map(nx, ny):
    """Create mapping from cushion indices to non-cushion indices."""
    nodemap = np.zeros((nx + 2, ny + 2), dtype=int)
    nodemap[1:nx + 1, 1:ny + 1] = np.arange(nx * ny).reshape((nx, ny))
    return nodemap[:, ::-1]


def _check_sources_receivers_inside_extent(srcs, recs, extent):
    """Validate that all sources and receivers lie within the model extent."""
    xmin, xmax, ymin, ymax = extent

    if not np.all((xmin <= recs[:, 0]) & (recs[:, 0] <= xmax) & (ymin <= recs[:, 1]) & (recs[:, 1] <= ymax)):
        raise InputError(
            msg=f"Input Error: One or more receiver lies outside of model extent: {extent}"
            "\nRemedy: adjust receiver locations and run again."
        )

    if not np.all((xmin <= srcs[:, 0]) & (srcs[:, 0] <= xmax) & (ymin <= srcs[:, 1]) & (srcs[:, 1] <= ymax)):
        raise InputError(
            msg=f"Input Error: One or more source lies outside of model extent: {extent}"
            "\nRemedy: adjust source locations and run again."
        )


def _check_requested_source_exists(tfieldsource, ns):
    """Warn if requested travel time field source does not exist."""
    if tfieldsource + 1 > ns:
        print(
            f"Error: Travel time field corresponds to source: {tfieldsource}\n"
            f"       but total number of sources is {ns}.\n"
            "       No travel time field will be calculated.\n"
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
    """Compute the L2 norm of a vector."""
    return np.sqrt(x.dot(x))


def normalise(x):
    """Normalize a vector to unit length."""
    return x / norm(x)


def png_to_model(pngfile, nx, ny, bg=1.0, sc=1.0):
    """Convert a PNG image to a velocity model."""
    png = Image.open(pngfile)
    png.load()
    grayscale = np.asarray(png.convert("L").resize((nx, ny)).transpose(Image.ROTATE_270))
    return sc * (bg + grayscale / 255.0)


def generate_surface_points(nPerSide, extent=(0, 1, 0, 1), surface=None, addCorners=True):
    """Generate points along the boundary of a rectangular extent."""
    if surface is None:
        surface = [True, True, True, True]

    out = []
    x = np.linspace(extent[0], extent[1], nPerSide + 2)[1:nPerSide + 1]
    y = np.linspace(extent[2], extent[3], nPerSide + 2)[1:nPerSide + 1]

    if surface[0]:
        out.extend([[extent[0], _y] for _y in y])
    if surface[1]:
        out.extend([[extent[1], _y] for _y in y])
    if surface[2]:
        out.extend([[_x, extent[2]] for _x in x])
    if surface[3]:
        out.extend([[_x, extent[3]] for _x in x])

    if addCorners:
        corners = [
            (surface[0] or surface[2], [extent[0], extent[2]]),
            (surface[0] or surface[3], [extent[0], extent[3]]),
            (surface[1] or surface[2], [extent[1], extent[2]]),
            (surface[1] or surface[3], [extent[1], extent[3]]),
        ]
        out.extend([pt for cond, pt in corners if cond])

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
    points=None,
    wfront=None,
    diced=True,
    dicex=8,
    dicey=8,
    cbarshrink=0.6,
    cbar=True,
    filename=None,
    reversedepth=False,
    points_size=1.0,
    aspect=None,
    ax=None,
    **wkwargs,
):
    """
    Plot 2D velocity or slowness field.

    Args:
        model: 2D velocity or slowness field on rectangular grid (nx, ny)
        paths: Ray paths to overlay on the model
        extent: Model extent [xmin, xmax, ymin, ymax]
        ax: Optional matplotlib axis object to plot onto
    """
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if cmap is None:
        cmap = plt.cm.RdBu

    plotmodel = create_diced_grid(model, extent=extent, dicex=dicex, dicey=dicey) if diced else model

    if reversedepth:
        extentr = [extent[0], extent[1], extent[3], extent[2]]
        im = ax.imshow(plotmodel.T, origin="upper", extent=extentr, aspect=aspect, cmap=cmap)
    else:
        im = ax.imshow(plotmodel.T, origin="lower", extent=extent, aspect=aspect, cmap=cmap)

    if paths is not None:
        if isinstance(paths, np.ndarray) and paths.ndim == 2 and paths.shape[1] == 4:
            paths = change_paths_format(paths)

        for i, p in enumerate(paths):
            cl = cline[i] if isinstance(cline, list) else cline
            lw = line[i] if isinstance(line, list) else line
            ax.plot(p[:, 0], p[:, 1], cl, lw=lw, alpha=alpha)

    if clim is not None:
        im.set_clim(clim)

    if title is not None:
        ax.set_title(title)

    if wfront is not None:
        nx, ny = wfront.shape
        X, Y = np.meshgrid(
            np.linspace(extent[0], extent[1], nx),
            np.linspace(extent[2], extent[3], ny),
        )
        ax.contour(X, Y, wfront.T, **wkwargs)

    if wfront is None and cbar:
        fig.colorbar(im, ax=ax, shrink=cbarshrink)

    if points is not None:
        ax.plot(points[:, 0], points[:, 1], 'bo', markersize=points_size)

    ax.set_xlim(extent[0], extent[1])
    if reversedepth:
        ax.set_ylim(extent[3], extent[2])
    else:
        ax.set_ylim(extent[2], extent[3])

    if created_ax:
        if filename is not None:
            fig.savefig(filename)
        plt.show()

def create_diced_grid(v, extent=None, dicex=8, dicey=8):
    """Create B-spline interpolated grid at higher resolution."""
    if extent is None:
        extent = [0.0, 1.0, 0.0, 1.0]
    nx, ny = v.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)

    # Reduce B-spline order if too few nodes
    kx = min(3, nx - 1)
    ky = min(3, ny - 1)

    rect = RectBivariateSpline(x, y, v, kx=kx, ky=ky)
    xx = np.linspace(extent[0], extent[1], dicex * nx)
    yy = np.linspace(extent[2], extent[3], dicey * ny)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    return rect.ev(X, Y)


def change_paths_format(paths):
    """Convert paths from (N, 4) format to (N, 2, 2) format."""
    return paths.reshape(-1, 2, 2)
