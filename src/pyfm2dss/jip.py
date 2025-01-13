#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:07:41 2021

A general set of routines to a Jacobian for linear inverse problem in 2D or 3D. 
This is done by integrating the product of a data kernel and a model basis function over the model domain.

Features:
- Data kernels and model bases specified by a user supplied class. (Examples included)
- 1D straight or curved line integrals of a 2D or 3D model basis.
- 2D/3D data kernels multiplied by a 2D/3D model basis function and integrated over the domain.
- Different methods of numerical integration allow choices to be made for trade off accuracy versus computational speed.
- Estimates of uncertainty for some cases.
- Special combinations of kernels and basis dealt with by specialized algorithms, e.g. straight ray based tomography in a celluar model.

@author:  Malcolm Sambridge
"""
import numpy as np
import scipy as scipy
import pyrr as pyrr
from scipy import integrate

# simps = integrate.simps
simps = integrate.simpson  # updated 16-09-24
tplquad = integrate.tplquad
dblquad = integrate.dblquad
quad = integrate.quad
from tqdm import tqdm


#
# calculate Jacobian by integrating all kernels x model bases using
# either simpsons or adaptive integration methods.
#
def calcjacobian(
    kernel,
    basis,
    mthd="simpson",
    epsrel=1.0e-3,
    returnerror=False,
    forceint=False,
    verbose=False,
    csc=False,
):
    """

    Compute all elements of the Jacobian matrix for a linear discrete inverse problem.

    Return the Jacobian (N_d x N_m) matrix and optionally its estimated error for a linear inverse problem
    with N_d data kernels and N_m model basis function. (N_d = kernel.nkernels; N_m = basis.nbases).

    The intention is that the user can change the data kernel and model basis by supplying their own classes,
    which align with the required structure.

    Example data kernels available include:

    - gravitykernel2D     : Implements gravity kernel about user supplied observation points in (x,z)
    - gravitykernel3D     : Implements gravity kernel about user supplied observation points in (x,y,z)
    - raykernel1D         : Implements seismic ray for straight or curved rays.

    Example model bases available include :

    - pixelbasis2D       : Implements 2D basis where the j-th basis has 1 is in the j-th cell and 0 elsewhere in 2D model.
    - voxelbasis3D       : Implements 3D basis where the j-th basis has 1 is in the j-th cell and 0 elsewhere in 3D model.
    - continuousbasis2D  : Implements 2D basis where the j-th basis is a continuous function of position (x,z) (dummy example).
    - continuousbasis3D  : Implements 3D basis where the j-th basis is a continuous function of position (x,y,z) (dummy example).


    Parameters
    ----------
    kernel : class
        This defines the nature of the forward problem and follows a data kernel class structure (see examples).
        It may represent a 1D, 2D, or 3D function representing the derivative of the forward problem with
        respect to the model. It has functions which evaluate the amplitude of the i-th kernel at position (i=1,...N_d)
        (x,z) or (x,y,z) in the model.

    basis : class
        The model basis class which defines the 2D or 3D model basis function class structure (see examples).
        It has functions which evaluate the amplitude of the j-th model basis (j=1,...N_m) at position (x,z) or
        (x,y,z) in the model. Has another function which defines the limits of the model domain in 2D or 3D.

    mthd : string
        'simpson' : means use scipy.integrate.simpson to perform numerical integration of the product of each data kernel
                    x model basis function over the model domain (default). This is fastest and least accurate.
                    Accuracy determined by grid discretization and integration limits supplied by basis class.
        'quad'    : means use scipy.integrate.tplquad or tblquad to perform numerical integration of the product of each
                    data kernel x model basis. This is slowest and most accurate. Accuracy controlled by integration limits
                    supplied by basis class and input parameter `epsrel`.

    epsrel : float, optional
        Relative tolerance of the inner 1-D integrals passed to scipy.integrate.tplquad or tblquad. Default is 1.e-3.

    returnerror : bool,optional
        Switch to return a matrix of same dimension as Jacobian containing error estimates for each value.
        Only applies if method==`quad`. Default is False.

    forceint : bool,optional
        Switch to force use of numerical integration method given by `mthd` even if specialized methods are available.
        For example if kernel is 1D and basis function is 2D pixel or 3D voxel, then J[i,j] equals the length of the
        i-th kernel (ray) in the j-th cell (2D or 3D) which can be found exactly and more efficently using a geometrical
        approach than through numerical integration. The default option is to ignore `mthd` in this case which is overridden
        by forceint=True. Mainly used to provide error checks on ray based method. Default is False.

    csc : bool,optional
        Switch to force J to be a csc_matrix, otherwise returns (dense) numpy array. Default is False.
        A sparse csc_matrix may be appropriate when the Jacobian is expected to be sparse, e.g. when kernels are 1D rays
        and model basis are pixels or voxels, but not when kernels are 1D rays and model basis is has global support, e.g. cosines.

    Returns
    -------
    J : float array of size N_d x N_m in scipy.sparse.csc_matrix format.
        The Jacobian matrix J[i,j] equals the integral of the i-th data kernel (supplied by kernel class) x j-th model basis
        function (supplied by basis class).

    Jerr : float array of size N_d x N_m, optional
        An estimate of the integration error in each element of the Jacobian. Requires mthd=`quad` and returnerror=True.
        Error estimate is also returned if forceint=True & kernel.type=`1Dstraightray` & basis.type = `3Dvoxel` or `2Dpixel`.

    """
    ndata = kernel.nkernels
    nbasis = basis.nbases
    if not csc:
        Jacobian = np.zeros([ndata, nbasis])
        if mthd == "quad":
            JacobianE = np.zeros([ndata, nbasis])

    if csc:  # output Jacobian in csc format
        k = 0
        indptr = np.zeros(nbasis + 1, dtype=int)
        Jindices = np.zeros(ndata * nbasis, dtype=int)
        Jdata = np.zeros(ndata * nbasis)
        if returnerror:
            JdataE = np.zeros(ndata * nbasis)
        for j in tqdm(range(nbasis)):  # loop over data
            indptr[j] = k
            for i in range(ndata):
                if verbose:
                    print(" Kernel function " + str(i) + " basis function " + str(j))
                if (
                    kernel.type == "1Dstraightray"
                    and (basis.type == "3Dvoxel" or basis.type == "2Dpixel")
                    and not forceint
                ):
                    r = raylengths(
                        i, j, kernel, basis
                    )  # special case for ray lengths in voxel cells.
                    if r != 0.0:
                        Jindices[k] = i
                        Jdata[k] = kernel.evaluate(i) * r
                        k += 1
                    # Jacobian[i,j] = kernel.evaluate(i)*raylengths(i,j,kernel,basis) # special case for ray lengths in voxel cells.
                elif mthd == "quad":
                    a, b = integrate_kernelandbasis(
                        i, j, kernel, basis, epsrel=epsrel, mthd="quad"
                    )
                    if a != 0.0:
                        Jindices[k] = i
                        Jdata[k] = a
                        JdataE[k] = b
                        k += 1
                elif mthd == "simpson":
                    r = integrate_kernelandbasis(i, j, kernel, basis, mthd="simpson")
                    if r != 0.0:
                        Jindices[k] = i
                        Jdata[k] = r
                        k += 1
        indptr[nbasis] = k
        Jacobian = scipy.sparse.csc_matrix(
            (Jdata[:k], Jindices[:k], indptr), shape=(ndata, nbasis)
        )
        if returnerror:
            Jacobian, scipy.sparse.csc_matrix(
                (JdataE[:k], Jindices[:k], indptr), shape=(ndata, nbasis)
            )
        # return indptr,Jindices,Jdata,k

    else:
        for i in tqdm(range(ndata)):  # loop over data
            for j in range(nbasis):
                if verbose:
                    print(" Kernel function " + str(i) + " basis function " + str(j))
                if (
                    kernel.type == "1Dstraightray"
                    and (basis.type == "3Dvoxel" or basis.type == "2Dpixel")
                    and not forceint
                ):
                    # print(' using line integral')
                    Jacobian[i, j] = kernel.evaluate(i) * raylengths(
                        i, j, kernel, basis
                    )  # special case for ray lengths in voxel cells.
                elif mthd == "quad":
                    Jacobian[i, j], JacobianE[i, j] = integrate_kernelandbasis(
                        i, j, kernel, basis, epsrel=epsrel, mthd="quad"
                    )
                elif mthd == "simpson":
                    Jacobian[i, j] = integrate_kernelandbasis(
                        i, j, kernel, basis, mthd="simpson"
                    )
    if returnerror:
        return Jacobian, JacobianE  # return Jacobian and estimated numerical error
    return Jacobian  # return Jacobian and estimated numerical error


def integrate_kernelandbasis(
    i, j, kernel, basis, epsrel=1.0e-3, mthd="simpson"
):  # Adaptive numerical integration

    if kernel.type == "1Dstraightray" or kernel.type == "1Dcurveray":
        if mthd == "simpson":
            return simps1D(i, j, kernel, basis)
        if mthd == "quad":
            return quad1D(i, j, kernel, basis, epsrel=epsrel)
    elif kernel.type == "2D":
        if mthd == "quad":
            return quad2D(i, j, kernel, basis, epsrel=epsrel)
        if mthd == "simpson":
            return simps2D(i, j, kernel, basis)
    elif kernel.type == "3D":
        if mthd == "quad":
            return quad3D(i, j, kernel, basis, epsrel=epsrel)
        if mthd == "simpson":
            return simps3D(i, j, kernel, basis)


def quad1D(
    i, j, kernel, basis, epsrel=1.0e-3
):  # Adaptive numerical integration along a ray
    f = lambda x: integrand1D(i, j, kernel, basis, x)
    return quad(f, 0.0, kernel.lengths[i], epsrel=epsrel)


def quad2D(i, j, kernel, basis, epsrel=1.0e-3):
    f = lambda z, x: integrand3D(i, j, kernel, basis, (x, z))
    xlim, zlim = basis.intlimits(j)  # set limits of integration for this basis function
    return integrate.dblquad(
        f, xlim[0], xlim[1], lambda x: zlim[0], lambda x: zlim[1], epsrel=epsrel
    )


def quad3D(i, j, kernel, basis, epsrel=1.0e-3):
    f = lambda z, y, x: integrand3D(i, j, kernel, basis, (x, y, z))
    xlim, ylim, zlim = basis.intlimits(
        j
    )  # set limits of integration for this basis function
    return integrate.tplquad(
        f,
        xlim[0],
        xlim[1],
        lambda x: ylim[0],
        lambda x: ylim[1],
        lambda x, y: zlim[0],
        lambda x, y: zlim[1],
        epsrel=epsrel,
    )


def simps1D(i, j, kernel, basis):  # Simpsons rule integration for 1D kernel
    l = np.linspace(0.0, kernel.lengths[i], basis.nline)
    ff = integrand1D(i, j, kernel, basis, l)
    return simps(ff, x=l)


def simps2D(i, j, kernel, basis):  # Simpsons rule integration for 2D kernel
    xlim, zlim = basis.intlimits(j)  # get limits of integration for this basis
    x = np.linspace(xlim[0], xlim[1], basis.nxi)
    z = np.linspace(zlim[0], zlim[1], basis.nzi)
    X, Z = np.meshgrid(x, z, indexing="ij")
    ff = integrand3D(i, j, kernel, basis, (X, Z))
    return simps(simps(ff, x=z), x=x)


def simps3D(i, j, kernel, basis):  # Simpsons rule integration for 3D kernel
    xlim, ylim, zlim = basis.intlimits(j)  # get limits of integration for this basis
    x = np.linspace(xlim[0], xlim[1], basis.nxi)
    y = np.linspace(ylim[0], ylim[1], basis.nyi)
    z = np.linspace(zlim[0], zlim[1], basis.nzi)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    ff = integrand3D(i, j, kernel, basis, (X, Y, Z))
    return simps(simps(simps(ff, x=z), x=y), x=x)


# Integrand for a 3D kernel in a 3D model
def integrand3D(
    i, j, kernel, basis, pos
):  # evaluates the integrand for use by integration routines
    if (
        basis.type == "3Dvoxel" or basis.type == "2Dpixel"
    ):  # if voxel basis it is assumed that x,y,z lies within the jth cell where basis value is one.
        return kernel.evaluate(i, pos)
    else:
        return kernel.evaluate(i, pos) * basis.evaluate(j, pos)


# Integrand for a 2D kernel in a 2D model
def integrand2D(
    i, j, kernel, basis, x, y
):  # evaluates the integrand for use by integration routines
    if (
        basis.type == "3Dvoxel"
    ):  # if voxel basis it is assumed that x,y,z lies within the jth cell where basis value is one.
        return kernel.evaluate(i, x, y)
    else:
        return kernel.evaluate(i, x, y) * basis.evaluate(j, x, y)


# Integrand for a 1D ray in either a 2D or 3D model
def integrand1D(
    i, j, kernel, basis, l
):  # evaluates the integrand for use by integration routines
    # x,y,z = kernel.position(i,l)
    pos = kernel.position(i, l)
    return basis.evaluate(j, pos)


def checkinsidecell(
    x, xlim, ylim, zlim
):  # check if point is inside cell (not on boundary)
    if x[0] <= xlim[0]:
        return False
    if x[0] >= xlim[1]:
        return False
    if x[1] <= ylim[0]:
        return False
    if x[1] >= ylim[1]:
        return False
    if x[2] <= zlim[0]:
        return False
    if x[2] >= zlim[1]:
        return False
    return True


# Calculate ray lengths in 2D pixel or 3D voxel model
def raylengths(
    i, j, kernel, basis
):  # Calculates lengths of rays in cellular grid in 2D or 3D

    if basis.type == "3Dvoxel":  # 3D case
        xlim, ylim, zlim = basis.intlimits(j)  # get cell boundaries this basis function
        raystart = kernel.paths[i][0]  # co-ordinates of endpoint of line
        rayend = kernel.paths[i][1]  # co-ordinates of endpoint of line
    elif basis.type == "2Dpixel":  # 2D case
        xlim, ylim = basis.intlimits(j)  # get cell boundaries this basis function
        rxy = kernel.paths[i][0]
        raystart = np.append(rxy[0:2], 0.5)  # co-ordinates of endpoint of line
        rxy = kernel.paths[i][1]
        rayend = np.append(rxy[0:2], 0.5)  # co-ordinates of endpoint of line
        zlim = np.array([0.0, 1.0])

    aabb = np.array(
        [[xlim[0], ylim[0], zlim[0]], [xlim[1], ylim[1], zlim[1]]]
    )  # cell defined by extremes
    line = np.array([raystart, rayend])  # raypath
    p0 = pyrr.geometric_tests.ray_intersect_aabb(
        pyrr.ray.create_from_line(line), aabb
    )  # intersection point

    if p0 is None:  # line does not intersect cell
        # print('ray does not intersect',i,j)
        return 0.0
    else:
        # deal with special cases of ray end points inside cells
        if checkinsidecell(raystart, xlim, ylim, zlim):
            if checkinsidecell(rayend, xlim, ylim, zlim):
                return np.linalg.norm(
                    raystart - rayend
                )  # both end points are inside cell return raylength
            return np.linalg.norm(
                raystart - p0
            )  # if start point is in cell return raylength
        if checkinsidecell(rayend, xlim, ylim, zlim):
            return np.linalg.norm(
                rayend - p0
            )  # if end point is in cell return raylength

        line = np.array(
            [rayend, raystart]
        )  # reverse raypath to find exit point from cell
        p1 = pyrr.geometric_tests.ray_intersect_aabb(
            pyrr.ray.create_from_line(line), aabb
        )  # intersection point
        if (
            p1 is None
        ):  # This case will arise when there is inconsistent results from pyrr package.
            # forward ray indicates intresection with cell but reversed ray does not.
            # Must be due to rounding error, so we ignore intersection for this case.
            return 0.0
        return np.linalg.norm(p1 - p0)  # return raylength in cell


def calcjacobian_nonsparse(
    kernel,
    basis,
    mthd="simpson",
    epsrel=1.0e-3,
    returnerror=False,
    forceint=False,
    verbose=False,
):
    """

    Compute all elements of the Jacobian matrix for a linear discrete inverse problem.

    Return the Jacobian (N_d x N_m) matrix and optionally its estimated error for a linear inverse problem
    with N_d data kernels and N_m model basis function. (N_d = kernel.nkernels; N_m = basis.nbases).

    The intention is that the user can change the data kernel and model basis by supplying their own classes,
    which align with the required structure.

    Example data kernels available include:

    - gravitykernel2D     : Implements gravity kernel about user supplied observation points in (x,z)
    - gravitykernel3D     : Implements gravity kernel about user supplied observation points in (x,y,z)
    - raykernel1D         : Implements seismic ray for straight or curved rays.

    Example model bases available include :

    - pixelbasis2D       : Implements 2D basis where the j-th basis has 1 is in the j-th cell and 0 elsewhere in 2D model.
    - voxelbasis3D       : Implements 3D basis where the j-th basis has 1 is in the j-th cell and 0 elsewhere in 3D model.
    - continuousbasis2D  : Implements 2D basis where the j-th basis is a continuous function of position (x,z) (dummy example).
    - continuousbasis3D  : Implements 3D basis where the j-th basis is a continuous function of position (x,y,z) (dummy example).


    Parameters
    ----------
    kernel : class
        This defines the nature of the forward problem and follows a data kernel class structure (see examples).
        It may represent a 1D, 2D, or 3D function representing the derivative of the forward problem with
        respect to the model. It has functions which evaluate the amplitude of the i-th kernel at position (i=1,...N_d)
        (x,z) or (x,y,z) in the model.

    basis : class
        The model basis class which defines the 2D or 3D model basis function class structure (see examples).
        It has functions which evaluate the amplitude of the j-th model basis (j=1,...N_m) at position (x,z) or
        (x,y,z) in the model. Has another function which defines the limits of the model domain in 2D or 3D.

    mthd : string
        'simpson' : means use scipy.integrate.simpson to perform numerical integration of the product of each data kernel
                    x model basis function over the model domain (default). This is fastest and least accurate.
                    Accuracy determined by grid discretization and integration limits supplied by basis class.
        'quad'    : means use scipy.integrate.tplquad or tblquad to perform numerical integration of the product of each
                    data kernel x model basis. This is slowest and most accurate. Accuracy controlled by integration limits
                    supplied by basis class and input parameter `epsrel`.

    epsrel : float, optional
        Relative tolerance of the inner 1-D integrals passed to scipy.integrate.tplquad or tblquad. Default is 1.e-3.

    returnerror : bool,optional
        Switch to return a matrix of same dimension as Jacobian containing error estimates for each value.
        Only applies if method==`quad`. Default is False.

    forceint : bool,optional
        Switch to force use of numerical integration method given by `mthd` even if specialized methods are available.
        For example if kernel is 1D and basis function is 2D pixel or 3D voxel, then J[i,j] equals the length of the
        i-th kernel (ray) in the j-th cell (2D or 3D) which can be found exactly and more efficently using a geometrical
        approach than through numerical integration. The default option is to ignore `mthd` in this case which is overridden
        by forceint=True. Mainly used to provide error checks on ray based method. Default is False.

    Returns
    -------
    J : float array of size N_d x N_m in scipy.sparse.csc_matrix format.
        The Jacobian matrix J[i,j] equals the integral of the i-th data kernel (supplied by kernel class) x j-th model basis
        function (supplied by basis class).

    Jerr : float array of size N_d x N_m, optional
        An estimate of the integration error in each element of the Jacobian. Requires mthd=`quad` and returnerror=True.
        Error estimate is also returned if forceint=True & kernel.type=`1Dstraightray` & basis.type = `3Dvoxel` or `2Dpixel`.

    """
    ndata = kernel.nkernels
    nbasis = basis.nbases
    Jacobian = np.zeros([ndata, nbasis])
    if mthd == "quad":
        JacobianE = np.zeros([ndata, nbasis])

    for i in tqdm(range(ndata)):  # loop over data
        for j in range(nbasis):
            if verbose:
                print(" Kernel function " + str(i) + " basis function " + str(j))
            if (
                kernel.type == "1Dstraightray"
                and (basis.type == "3Dvoxel" or basis.type == "2Dpixel")
                and not forceint
            ):
                # print(' using line integral')
                Jacobian[i, j] = kernel.evaluate(i) * raylengths(
                    i, j, kernel, basis
                )  # special case for ray lengths in voxel cells.
            elif mthd == "quad":
                Jacobian[i, j], JacobianE[i, j] = integrate_kernelandbasis(
                    i, j, kernel, basis, epsrel=epsrel, mthd="quad"
                )
            elif mthd == "simpson":
                Jacobian[i, j] = integrate_kernelandbasis(
                    i, j, kernel, basis, mthd="simpson"
                )
    if returnerror:
        return Jacobian, JacobianE  # return Jacobian and estimated numerical error
    return Jacobian  # return Jacobian and estimated numerical error
