#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 07:31:45 2021

@author: Malcolm Sambridge
"""
import numpy as np
import pickle

# Define available basis classes


class PixelBasis2D:  # Model basis class
    """
    A 2D Model pixel basis object.
    A class that is used by various routines to calculate Jacobians through numerical integration.

    """

    def __init__(
        self, xg, zg, npoints=[50, 50, 1000], outer="x"
    ):  # receives linear grid dimension
        self.type = "2Dpixel"
        self.name = "2Dpixel"
        self.xg = xg  # X-grid for voxel model
        self.zg = zg  # Y-grid for voxel model
        self.nx = len(xg) - 1  # Number of cells in X-direction
        self.nz = len(zg) - 1  # Number of cells in Y-direction
        self.xmin = xg[0]  # X-min of model
        self.xmax = xg[-1]  # X-max of model
        self.zmin = zg[0]  # Y-min of model
        self.zmax = zg[-1]  # Y-max of model
        self.outer = outer  # axis for outer loop of basis grid (for indexing)
        self.nbases = self.nx * self.nz  # Number of basis functions in voxel model

        # X-integration grid used in each cell of this basis
        # (only used if kernel is 3D and integration mthd is 'simpson')
        self.nxi = npoints[0]
        # Y-integration grid used in each cell of this basis
        # (only used if kernel is 3D and integration mthd is 'simpson')
        self.nzi = npoints[1]
        # Number of integration points along 1D ray within cell
        # (only used if kernel is a 1D and integration method `simpson')
        self.nline = npoints[2]

    def evaluate(self, j, pos):
        """evaluate the ith model basis function at location (x,y)"""
        if j < 0 or j >= self.nbases:
            raise BasisError(j)
        x, z = pos
        ix, iz = self.convert_index(j)  # convert j basis to 2D model basis index
        if np.isscalar(x):
            b = 0.0  # pixel basis value
            if (
                (x >= self.xg[ix])
                & (x < self.xg[ix + 1])
                & (z >= self.zg[iz])
                & (z < self.zg[iz + 1])
            ):
                b = 1.0
            if (ix == self.nx - 1 and x == self.xg[-1]) or (
                iz == self.nz - 1 and z == self.zg[-1]
            ):
                b = 1.0  # Correct upper boundary
        else:
            b = np.zeros_like(x)
            b[
                (x >= self.xg[ix])
                & (x < self.xg[ix + 1])
                & (z >= self.zg[iz])
                & (z < self.zg[iz + 1])
            ] = 1.0
            b[(ix == self.nx - 1 and x == self.xg[-1])] = 1.0
            b[(iz == self.nz - 1 and z == self.zg[-1])] = 1.0
        return b

    def convert_index(self, i):
        """
        Convert single 1D array index to 2D array index using inner loop over X and outer loop over Y
        """
        if i < 0 or i >= self.nbases:
            raise BasisError(i)
        if self.outer == "x":
            zindex = int(i % self.nz)
            xindex = int((i / self.nz) % self.nx)
        else:
            xindex = int(i % self.nx)
            zindex = int((i / self.nx) % self.nz)
        return xindex, zindex

    def iconvert_index(self, ix, iz):
        """
        Convert single 2D array index to 1D array index inner loop over X and outer loop over Y
        """
        if ix < 0 or ix >= self.nx:
            raise BasisError(ix)
        if iz < 0 or iz >= self.nz:
            raise BasisError(iz)
        if self.outer == "x":
            return ix * self.nz + iz  # iz is the inner loop and ix the outer loop
        else:
            return iz * self.nx + ix  # ix is the inner loop and iz the outer loop

    def int_limits(self, j):
        """calculates the limits of integration for given basis function"""
        ix, iz = self.convert_index(j)  # convert j basis to 2D model basis index
        # For voxel basis we return the limits of the jth cell (because jth basis is zero elsewhere)
        return [self.xg[ix], self.xg[ix + 1]], [self.zg[iz], self.zg[iz + 1]]


class BasisError(Exception):
    """Raised when input model basis id does not exist in kernel definition"""

    def __init__(self, cset=[]):
        super().__init__("\n Basis id " + str(cset) + " not recognized. \n")


class CosineBasis2D:  # Model basis class
    """
    A 2D Model cosine basis object.
    A class that is used by various routines to calculate Jacobians through numerical integration.

    """

    def __init__(
        self, x0, x1, z0, z1, nx, ny, npoints=[120, 120, 200]
    ):  # receives linear grid dimension
        self.type = "2D"
        self.name = "2Dcosine"
        self.xmin = x0  # X-min of model
        self.xmax = x1  # X-max of model
        self.zmin = z0  # Z-min of model
        self.zmax = z1  # Z-max of model
        self.nx = nx  # set number of bases in X-direction
        self.nz = ny  # set number of bases in Z-direction
        self.nbases = self.nx * self.nz  # Total number of basis functions in 2D model

        self.nxi = npoints[0]
        # X-integration grid used in each cell of this basis
        # (only used if kernel is 3D and integration mthd is 'simpson')
        self.nzi = npoints[1]
        # Y-integration grid used in each cell of this basis
        # (only used if kernel is 3D and integration mthd is 'simpson')
        self.nline = npoints[2]
        # Number of integration points along 1D ray within cell
        # (only used if kernel is a 1D and integration method `simpson')
        self.Lx = self.xmax - self.xmin  # Set maximum X wavelength in model
        self.Lz = self.zmax - self.zmin  # Set maximum Z wavelength in model
        self.norm = np.sqrt(self.Lx * self.Lz)
        self.area = self.Lx * self.Lz  # set area of domain

    def evaluate(self, j, pos):  # evaluate the ith data kernel at location (x,z)
        x, z = pos
        ix, iz = self.convert_index(
            j
        )  # convert from single index to pair of ix,iz indices
        b = np.cos(np.pi * ix * (x - self.xmin) / (self.Lx)) * np.cos(
            np.pi * iz * (z - self.zmin) / (self.Lz)
        )  # evaluate jth basis function at input positions
        fx, fz = np.sqrt(2.0), np.sqrt(2.0)
        if ix == 0:
            fx = 1.0
        if iz == 0:
            fz = 1.0
        return b * fx * fz / self.norm

    def convert_index(self, i):
        """
        Convert single 1D array index to 2D array index using inner loop over X and outer loop over Y
        """
        if i < 0 or i >= self.nbases:
            raise BasisError(i)

        xindex = int(i % self.nx)
        zindex = int((i / self.nx) % self.nz)
        return xindex, zindex

    def iconvert_index(self, ix, iz):
        """
        Convert single 2D array index to 1D array index inner loop over X and outer loop over Y
        """
        if ix < 0 or ix >= self.nx:
            raise BasisError(ix)
        if iz < 0 or iz >= self.nz:
            raise BasisError(iz)
        return iz * self.nx + ix  # ix is the inner loop and iz the outer loop

    def int_limits(self, j):
        """calculates the limits of integration for given basis function"""
        return [self.xmin, self.xmax], [
            self.zmin,
            self.zmax,
        ]  # Here we choose whole model

    def fit_coefficients(self, target, A=None, returnmatrix=False):
        """
        fitcoefficients - fits basis coefficients to given field of function values at arbitrary resolution.

        Inputs:
             target, ndarray(nx,nz) -               : input function field to fit model basis to
             A, ndarray(nx*nz,self.nx*self.nz) -    : transformation matrix of basis coefficients at each point in inpit field.
                                                      A[i,j] is the j basis function evaluated at ith point in the function field,
                                                      where both indices correspond to flattened arrays.
                                                      A is calculated if A=None, and returned if returnmaxtrix = True.
        Returns:
            coeff, ndarray(self.nx,self.ny)         : best fit basis coefficients
        """
        nx, nz = target.shape
        if type(A) is not np.ndarray:
            A = np.zeros((self.nbases, nx * nz))
            for j in range(self.nbases):
                A[j] = self.get_basis_image(j, nx=nx, nz=nz).flatten()
        coeff, res, rank, s = np.linalg.lstsq(
            A.T, target.flatten(), rcond=None
        )  # fit basis coefficients to input field
        if returnmatrix:
            return coeff.reshape(self.nx, self.nz), A
        return coeff.reshape(self.nx, self.nz)

    def get_basis_image(self, j, nx=None, nz=None):
        """returns 2D image of model at chosen resolution"""
        if nx is None:
            nx = self.nx
        if nz is None:
            nz = self.nz
        dx, dz = (self.xmax - self.xmin) / nx, (self.zmax - self.zmin) / nz
        Zm, Xm = np.meshgrid(
            np.linspace(self.zmin + dz / 2, self.zmax - dz / 2, nz),
            np.linspace(self.xmin + dx / 2.0, self.xmax - dx / 2, nx),
        )
        return self.evaluate(j, (Xm, Zm))

    # Define available data kernel classes


class RayKernel1D:  # A linear data kernel class
    """
    Data kernel object.

    A class that is used by various routines to calculate Jacobians through numerical integration.

    """

    def __init__(self, paths):  # receives the end points of rays.
        self.nkernels = len(paths)
        self.paths = paths
        self.constant = 1.0  # kernel is a constant along the ray (typically 1.0)
        d = np.zeros(self.nkernels)
        if isinstance(paths, list):
            self.type = "1Dcurveray"
            dcumsum = []
            for k, p in enumerate(paths):
                dl = p[1:] - p[:-1]  # co-ordinates of ray segment end points
                dls = np.linalg.norm(dl, axis=1)  # lengths of each ray segment
                dcumsum.append(
                    np.array([0.0, *np.cumsum(dls)])
                )  # cummulative lengths of each ray segment
                d[k] = np.sum(dls)  # lengths of each ray
            self.cumlengths = dcumsum
        else:
            self.type = "1Dstraightray"  # This id indicates a straight ray (use `1D' if ray is not straight line)
            for k, p in enumerate(paths):
                dl = p[1:] - p[:-1]  # co-ordinates of ray segment end points
                dls = np.linalg.norm(dl, axis=1)  # lengths of each ray segment
                d[k] = np.sum(dls)  # lengths of each ray
        self.lengths = d

    def evaluate(self, i):  # evaluate the ith data kernel at location (x,y,z)
        if i < 0 or i >= self.nkernels:
            raise KernelOBSError(i)

        return self.constant  # Return constant along ray for integration

    def position(self, i, l):
        """
        find (x,y,z) position of ith data kernel at length l along ray.

        This example is for a straight seismic ray (self.type = '1Dstraightray').
        For a curved ray (self.type = '1D') this is where one would specify (x,y,z) as a function of length l.
        """
        if i < 0 or i >= self.nkernels:
            raise KernelOBSError(i)
        # if(l <= 0): return self.paths[i][0]
        # if(l >= self.lengths[i]): return self.paths[i][-1]
        if self.type == "1Dstraightray":
            alpha = l / self.lengths[i]
            a0 = self.paths[i][0]
            a1 = self.paths[i][1]
            return (1.0 - alpha) * a0[:, np.newaxis] + alpha * a1[
                :, np.newaxis
            ]  # here we use a straight line between endpoints of locations
        else:
            j = np.clip(
                np.searchsorted(self.cumlengths[i], l), 1, len(self.cumlengths[i]) - 1
            )
            # print(i,self.lengths[i],l,j,self.cumlengths[i])
            d0 = self.cumlengths[i][j - 1]
            d1 = self.cumlengths[i][j]
            dl = d1 - d0
            a0 = self.paths[i][j - 1]
            a1 = self.paths[i][j]
            alpha = (l - d0) / dl
            out = (1.0 - alpha[:, np.newaxis]) * a0 + alpha[
                :, np.newaxis
            ] * a1  # interpolate along segment for location
            return out.T


class Error(Exception):
    """Base class for other exceptions"""

    pass


class KernelOBSError(Exception):
    """Raised when input kernel id does not exist in kernel definition"""

    def __init__(self, cset=[]):
        super().__init__("\n Kernel id " + str(cset) + " not recognized. \n")


# Define utility routines for 2D tomography


def randomrays2Dbox(Nrays, x, y, seed=61254557, filename=""):
    """Generate random rays across 2D model"""
    if filename == "":  # generate ray endpoints randomly on side of given box
        np.random.seed(seed)
        sides = np.zeros((Nrays, 2), dtype=int)
        for i in range(Nrays):
            sides[i] = np.random.choice(4, replace=False, size=2)
        paths2D = np.zeros((Nrays, 2, 2))
        for i in range(Nrays):
            paths2D[i, 0] = _choose_endpoint(sides[i, 0], x, y)
            paths2D[i, 1] = _choose_endpoint(sides[i, 1], x, y)
    else:  # read ray end points from pickle file
        with open(filename, "rb") as f:
            paths2D = pickle.load(f)
    return paths2D


def _choose_endpoint(side, x, y):  # choose random point on boundary of 2D box
    if side == 0:
        # choose endpoint randomly along xmin side
        return [
            x[0],
            y[0] + (y[-1] - y[0]) * np.random.random(),
        ]
    elif side == 1:
        # choose endpoint randomly along xmax side
        return [
            x[-1],
            y[0] + (y[-1] - y[0]) * np.random.random(),
        ]
    elif side == 2:
        # choose endpoint randomly along ymin side
        return [
            x[0] + (x[-1] - x[0]) * np.random.random(),
            y[0],
        ]
    else:
        # choose endpoint randomly along ymax side
        return [
            x[0] + (x[-1] - x[0]) * np.random.random(),
            y[-1],
        ]


def _generate_center_coordinates(l_x):
    l_x = float(l_x)
    X, Y = np.mgrid[:l_x, :l_x]
    center = l_x / 2.0
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y
