import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .fastmarching import FastMarchingMethod
from . import bases as base
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csr_matrix
import faulthandler

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


class WaveTracker:

    def __init__(self):
        self.fmm = FastMarchingMethod()

    def set_times(self, t):
        self.ttimes = t.copy()

    def set_paths(self, p):
        self.paths = p.copy()

    def set_frechet(self, f):
        self.frechet = f.copy()

    def set_tfield(self, w, source):
        self.tfield = w[source].copy()
        self.tfield = self.tfield[
            :, ::-1
        ]  # adjust y axis of travel time field as it is provided in reverse ordered.
        self.tfieldsource = source

    def calc_wavefronts(
        self,
        v,
        recs,
        srcs,
        verbose=False,
        paths=False,
        frechet=False,
        times=True,
        tfieldsource=-1,
        sourcegridrefine=True,
        sourcedicelevel=5,
        sourcegridsize=10,
        earthradius=6371.0,
        schemeorder=1,
        nbsize=0.5,
        degrees=False,
        velocityderiv=False,
        extent=[0.0, 1.0, 0.0, 1.0],
        dicex=8,
        dicey=8,
    ):
        """

        A function to perform 2D Fast Marching of wavefronts from sources in a 2D velocity model.

        Inputs:
            v, ndarray(nx,ny)          : coefficients of velocity field in 2D grid
            recs, ndarray(nr,2)        : receiver locations (x,y)
            srcs, ndarray(ns,2)        : source locations (x,y)
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
            Internally variables are converted to np.float32 to be consistent with Fortran code fm2dss.f90.

        """

        recs = recs.reshape(-1, 2)  # ensure receiver array is 2D and float32
        srcs = srcs.reshape(-1, 2)  # ensure source array is 2D and float32
        # check sources and receives inside extent

        ns = len(srcs)

        if (
            tfieldsource + 1 > ns
        ):  # source requested for travel time field does not exist
            print(
                "Error: Travel time field corresponds to source:",
                tfieldsource,
                "\n",
                "      but total number of sources is",
                len(srcs),
                "\n       No travel time field will be calculated.\n",
            )

        # fmst expects input spatial co-ordinates in degrees and velocities in kms/s so we adjust (unless degrees=True)

        kms2deg = 180.0 / (earthradius * np.pi)

        lpaths = 0  # Write out raypaths (<0=all,0=no,>0=source id)
        if paths:
            lpaths = -1  # only allow all or none

        lttimes = 0  # int to calculate travel times (y=1,n=0)
        if times:
            lttimes = 1

        lfrechet = 0  # bool to calculate Frechet derivatives of travel times w.r.t. slownesses (0=no,1=yes)
        if frechet:
            lfrechet = 1

        tsource = 0
        if tfieldsource >= 0:
            tsource = 1  # int to calculate travel fields (0=no,1=all)

        self.fmm.set_solver_options(
            np.int32(dicex),  # set solver options
            np.int32(dicey),
            np.int32(sourcegridrefine),
            np.int32(sourcedicelevel),
            np.int32(sourcegridsize),
            np.float32(earthradius),
            np.int32(schemeorder),
            np.float32(nbsize),
            np.int32(lttimes),
            np.int32(lfrechet),
            np.int32(tsource),
            np.int32(lpaths),
        )

        scx = np.float32(srcs[:, 0])
        scy = np.float32(srcs[:, 1])

        self.fmm.set_sources(
            scy, scx
        )  # set sources (ordering inherited from fm2dss.f90)

        rcx = np.float32(recs[:, 0])
        rcy = np.float32(recs[:, 1])
        self.fmm.set_receivers(rcy, rcx)  # set receivers

        nvx, nvy, dlat, dlong, vc = self.build_velocity_grid(
            v, extent
        )  # add cushion layer to velocity model and get parameters

        nvx = np.int32(nvx)
        nvy = np.int32(nvy)
        extent = np.array(extent, dtype=np.float32)
        dlat = np.float32(dlat)
        dlong = np.float32(dlong)
        vc = vc.astype(np.float32)

        self.fmm.set_velocity_model(nvy, nvx, extent[3], extent[0], dlat, dlong, vc)

        srs = np.ones(
            (len(recs), len(srcs)), dtype=np.int32
        )  # set up time calculation between all sources and receivers

        self.fmm.set_source_receiver_associations(srs)

        self.fmm.allocate_result_arrays()  # allocate memory for Fortran arrays

        self.fmm.track()  # run fmst wavefront tracker code

        # collect results
        if times:
            ttimes = self.fmm.get_traveltimes()
            if not degrees:
                ttimes *= (
                    kms2deg  # adjust travel times because inputs are not in degrees
                )

        if paths:
            raypaths = self.fmm.get_raypaths()

        if frechet:
            # frechetvals = read_fmst_frechet(wdir+'/'+ffilename,noncushion,nodemap)
            frechetvals = self.fmm.get_frechet_derivatives()
            if not degrees:
                frechetvals *= (
                    kms2deg  # adjust travel times because inputs are not in degrees
                )

            # the frechet matrix returned in in csr format and has two layers of cushion nodes surrounding the (nx,ny) grid
            F = frechetvals.toarray()  # unpack csr format
            nrays = np.shape(F)[0]  # number of raypaths
            nx, ny = v.shape  # shape of non-cushion velcoity model
            F = F[:, self.noncushion.flatten()].reshape(
                (nrays, nx, ny)
            )  # remove cushion nodes and reshape to (nx,ny)
            F = F[
                :, :, ::-1
            ]  # reverse y order, because it seems to be returned in reverse order (cf. ttfield array)
            frechetvals = csr_matrix(
                F.reshape((nrays, nx * ny))
            )  # reformat as a sparse CSR matrix
            if (
                not velocityderiv
            ):  # adjust derivatives to be of velocites rather than slownesses (default)
                x2 = -(v * v).reshape(-1)
                frechetvals = frechetvals.multiply(x2)

        if tfieldsource >= 0:
            tfieldvals = self.fmm.get_traveltime_fields()
            if not degrees:
                tfieldvals *= (
                    kms2deg  # adjust travel times because inputs are not in degrees
                )

        #   add required information to class instances

        if times:
            self.set_times(ttimes)
        if paths:
            self.set_paths(raypaths)
        if frechet:
            self.set_frechet(frechetvals)
        if tfieldsource > -1:
            self.set_tfield(
                tfieldvals, tfieldsource
            )  # set traveltime field and source id

        self.fmm.deallocate_result_arrays()

        return

    def build_velocity_grid(
        self, v, extent
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
        vc[0, 0], vc[0, -1], vc[-1, 0], vc[-1, -1] = (
            v[0, 0],
            v[0, -1],
            v[-1, 0],
            v[-1, -1],
        )
        vc = vc[:, ::-1]

        self.noncushion = noncushion
        self.nodemap = nodemap.flatten()

        return nx, ny, dlat, dlong, vc


class GridModel:  # This is for the original regular model grid (without using the basis.py package)

    def __init__(self, velocities, extent=(0, 1, 0, 1)):
        self.nx, self.ny = velocities.shape
        self.velocities = velocities
        self.xmin, self.xmax, self.ymin, self.ymax = extent
        self.xx = np.linspace(self.xmin, self.xmax, self.nx + 1)
        self.yy = np.linspace(self.ymin, self.ymax, self.ny + 1)
        # self.dicex = dicex
        # self.dicey = dicey
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

    def generate_surface_points(
        self,
        nPerSide,
        extent=(0, 1, 0, 1),
        surface=[True, True, True, True],
        addCorners=True,
    ):
        out = []
        if surface[0]:
            out += [
                [extent[0], x]
                for x in np.linspace(extent[2], extent[3], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
        if surface[1]:
            out += [
                [extent[1], x]
                for x in np.linspace(extent[2], extent[3], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
        if surface[2]:
            out += [
                [x, extent[2]]
                for x in np.linspace(extent[0], extent[1], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
        if surface[3]:
            out += [
                [x, extent[3]]
                for x in np.linspace(extent[0], extent[1], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
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
            # self.dicex = dicex
            # self.dicey = dicey
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
                    return (
                        self.coeffs.copy()
                    )  # because coefficients must be for velocities
                return (
                    self.vref + self.coeffs.copy()
                )  # because coefficients must be for velocities
            else:
                if returncoeff:
                    return (
                        1.0 / self.coeffs
                    )  # because coefficients must be for slownesses
                return 1.0 / (
                    self.sref + self.coeffs
                )  # because coefficients must be for slownesses
        else:  # non pixel basis
            if self.coeff_type == "velocities":
                if returncoeff:
                    return (
                        self.coeffs.copy()
                    )  # coefficients are velocities and so we return coefficients
                return self.vref + self.get_image(
                    nx=nx, ny=ny
                )  # return a velocity field evaluated from basis summation
            else:
                if returncoeff:
                    return (
                        self.fit_coefficients_s2v()
                    )  # coefficients are slownesses and we need to find equivalent velocity coefficients
                return 1.0 / (
                    self.sref + self.get_image(nx=nx, ny=ny)
                )  # coefficients are slownesses and we must return a velocity field

    def basis_transform_matrix(self):
        if not self.A_calc:
            A = np.zeros((self.basis.nbases, self.nx * self.ny))
            for j in range(self.basis.nbases):
                A[j] = self.get_basis_image(j, nx=self.nx, ny=self.ny).flatten()
            self.A = A
            self.A_calc = True
        return self.A

    def fit_coefficientes_v2s(
        self, nx=None, ny=None
    ):  # calculate slowness coefficients that correspond to a given set of velocity coefficients in model basis
        if nx == None:
            nx = self.nx
        if ny == None:
            ny = self.ny
        vtarget = self.get_velocity(nx=nx, ny=ny) - self.vref
        if not self.A_calc:
            A = self.basis_transform_matrix()
            self.A = A
        slowcoeff, res, rank, s = np.linalg.lstsq(
            self.A.T, 1.0 / vtarget.flatten(), rcond=None
        )  # fit slownesses coefficients to slowness field
        return slowcoeff.reshape(self.nx, self.ny)

    def fit_coefficients_s2v(
        self, nx=None, ny=None
    ):  # calculate velocity coefficients that correspond to a given set of slowness coefficients in model basis
        if nx == None:
            nx = self.nx
        if ny == None:
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

    def convert_pixel_vel_2_basis_slow(
        self, v
    ):  # convert velocity model in pixel basis to equivalent model as slowness coefficients in model basis
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

    def convert_pixel_vel_2_basis_vel(
        self, v
    ):  # convert velocity model in pixel basis to equivalent model as velocity coefficients in model basis
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
        """With no arguments this will return a slowness field.
        If bases are 2D pixels the keyword returncoeff is ignored
        If bases are not pixels and returncoeff is False, then a slowness field is evaluated and returned at (nx,ny)
        If bases are not 2D pixels and returncoeff is True, then slowness basis coefficients are returned.
        Default values of (nx,ny) are determined by .getImage() and are the input resolution of the model.
        """
        if self.basis_type == "2Dpixel":
            if self.coeff_type == "velocities":
                if returncoeff:
                    return (
                        1.0 / (self.vref + self.coeffs) - self.sref
                    )  # because coefficients are velocities
                return 1.0 / (
                    self.vref + self.coeffs
                )  # because coefficients are velocities
            else:
                if returncoeff:
                    return self.coeffs.copy()  # because coefficients are slownesses
                return (
                    self.sref + self.coeffs.copy()
                )  # because coefficients are slownesses
        else:
            if self.coeff_type == "velocities":
                if returncoeff:
                    return (
                        self.fit_coefficientes_v2s()
                    )  # we need to find slowness coefficients from velocity coefficients here
                return 1.0 / (
                    self.vref + self.get_image(nx=nx, ny=ny)
                )  # return a slowness field after summation of velocity bases
            else:
                if returncoeff:
                    return (
                        self.coeffs.copy()
                    )  # coefficients are slownesses and so we return coefficients
                return self.sref + self.get_image(
                    nx=nx, ny=ny
                )  # return a slowness field after summation of slowness bases

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
        if nx == None:
            nx = self.nx
        if ny == None:
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
        if nx == None:
            nx = self.nx
        if ny == None:
            ny = self.ny
        dx, dy = (self.xmax - self.xmin) / nx, (self.ymax - self.ymin) / ny
        Ym, Xm = np.meshgrid(
            np.linspace(self.ymin + dy / 2, self.ymax - dy / 2, ny),
            np.linspace(self.xmin + dx / 2.0, self.xmax - dx / 2, nx),
        )
        return self.basis.evaluate(j, (Xm, Ym))

    # --------------------------------------------------------------------------------------------
    # Other utility functions
    #
    # --------------------------------------------------------------------------------------------

    def norm(self, x):
        return np.sqrt(x.dot(x))

    def normalise(self, x):
        return x / self.norm(x)

    def png_to_model(self, pngfile, nx, ny, bg=1.0, sc=1.0):
        png = Image.open(pngfile)
        png.load()
        model = sc * (
            bg
            + np.asarray(png.convert("L").resize((nx, ny)).transpose(Image.ROTATE_270))
            / 255.0
        )
        return model

    def generate_surface_points(
        self,
        nPerSide,
        extent=(0, 1, 0, 1),
        surface=[True, True, True, True],
        addCorners=True,
    ):
        out = []
        if surface[0]:
            out += [
                [extent[0], x]
                for x in np.linspace(extent[2], extent[3], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
        if surface[1]:
            out += [
                [extent[1], x]
                for x in np.linspace(extent[2], extent[3], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
        if surface[2]:
            out += [
                [x, extent[2]]
                for x in np.linspace(extent[0], extent[1], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
        if surface[3]:
            out += [
                [x, extent[3]]
                for x in np.linspace(extent[0], extent[1], nPerSide + 2)[
                    1 : nPerSide + 1
                ]
            ]
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


class Plot:  # This is a set of plotting routines to display 2D velocity models and optionally raypaths and wavefronts on top.
    """

    A model class containing plot routines for display of 2D velocity/slowness models and optionally raypaths and wavefronts on top.

    """

    def __init__(self):
        pass

    def display_model(
        self,
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
            plotmodel = self.diced_grid(model, extent=extent, dicex=dicex, dicey=dicey)

        plt.imshow(plotmodel.T, origin="lower", extent=extent, cmap=cmap)

        if paths is not None:
            if isinstance(paths, np.ndarray) and paths.ndim == 2:
                if paths.shape[1] == 4:  # we have paths from xrt.tracer so adjust
                    paths = self.change_paths_format(paths)

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
            plt.contour(
                X, Y, wfront.T, **wkwargs
            )  # Negative contours default to dashed.

        if wfront is None and cbar:
            plt.colorbar(shrink=cbarshrink)

        if filename is not None:
            plt.savefig(filename)

        plt.show()

    def diced_grid(self, v, extent=[0.0, 1.0, 0.0, 1.0], dicex=8, dicey=8):
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

    def change_paths_format(self, paths):
        p = np.zeros((len(paths), 2, 2))
        for i in range(len(paths)):
            p[i, 0, 0] = paths[i, 0]
            p[i, 0, 1] = paths[i, 1]
            p[i, 1, 0] = paths[i, 2]
            p[i, 1, 1] = paths[i, 3]
        return p
