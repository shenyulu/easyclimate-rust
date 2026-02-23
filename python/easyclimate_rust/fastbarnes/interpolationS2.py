# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

"""
Module that provides two different Barnes interpolation algorithms acting on
the unit sphere S^2 and thus using the spherical distance metric.
To attain competitive performance, the code is written using Numba's
just-in-time compiler and thus has to use the respective programming idiom,
which is sometimes not straightforward to read at a first glance. Allocated
memory is as far as possible reused in order to reduce the workload imposed
on the garbage collector.

Created on Sat May 14 2022, 20:49:17
@author: Bruno Zürcher
"""

from math import exp, pi
import numpy as np

from numba import njit

from . import interpolation
from .util import lambert_conformal


###############################################################################

def barnes_S2(
        pts, val, sigma, x0, step, size,
        method='optimized_convolution_S2', num_iter=4, max_dist=3.5, resample=True,
        lambert_proj=None, lambert_grid=None, auto_proj=True):
    """
    Computes the Barnes interpolation for observation values `val` taken at sample
    points `pts` using Gaussian weights for the width parameter `sigma`.
    The underlying grid is embedded on the unit sphere S^2 and thus inherits the
    spherical distance measure (taken in degrees). The grid is given by the start
    point `x0`, regular grid steps `step` and extension `size`.

    Parameters
    ----------
    pts : numpy ndarray
        A 2-dimensional array of size N x 2 containing the x- and y-coordinates
        (or if you like the longitude/latitude) of the N sample points.
    val : numpy ndarray
        A 1-dimensional array of size N containing the N observation values.
    sigma : float
        The Gaussian width parameter to be used.
    x0 : numpy ndarray
        A 1-dimensional array of size 2 containing the coordinates of the
        start point of the grid to be used.
    step : float
        The regular grid point distance.
    size : tuple of 2 int values
        The extension of the grid in x- and y-direction.
    method : {'optimized_convolution_S2', 'naive_S2'}
        Designates the Barnes interpolation method to be used. The possible
        implementations that can be chosen are 'naive_S2' for the straightforward
        implementation (algorithm A from the paper) with an algorithmic complexity
        of O(N x W x H).
        The choice 'optimized_convolution_S2' implements the optimized algorithm B
        specified in the paper by appending tail values to the rectangular kernel.
        The latter algorithm has a reduced complexity of O(N + W x H).
        The default is 'optimized_convolution_S2'.
    num_iter : int, optional
        The number of performed self-convolutions of the underlying rect-kernel.
        Applies only if method is 'optimized_convolution_S2'.
        The default is 4.
    max_dist : float, optional
        The maximum distance between a grid point and the next sample point for which
        the Barnes interpolation is still calculated. Specified in sigma distances.
        The default is 3.5, i.e. the maximum distance is 3.5 * sigma.
    resample : bool, optional
        Specifies whether to resample Lambert grid field to lonlat grid.
        Applies only if method is 'optimized_convolution_S2'.
        The default is True.
    lambert_proj : tuple, optional
        Optional Lambert projection constants tuple as returned by
        `lambert_conformal.create_proj(center_lon, center_lat, lat1, lat2)`.
        If omitted and `auto_proj` is True, the projection is inferred from the
        target lon-lat grid.
    lambert_grid : tuple, optional
        Optional Lambert grid definition `(lam_x0, lam_size)` where `lam_x0` is
        a length-2 array-like and `lam_size` a length-2 tuple of ints.
        If omitted and `auto_proj` is True, the Lambert grid is inferred from
        the target lon-lat grid.
    auto_proj : bool, optional
        If True, infer missing Lambert projection and/or Lambert grid from
        `x0`, `step` and `size`. The default is True.

    Returns
    -------
    numpy ndarray
        A 2-dimensional array containing the resulting field of the performed
        Barnes interpolation.
    """    
    # perform simplified argument checking
    dim = pts.shape[1]

    # since we will modify the input array val in method _normalize_values(), we store a copy of it
    val = val.copy()

    # check sigma
    if isinstance(sigma, (list, tuple, np.ndarray)):
        if len(sigma) != dim:
            raise RuntimeError('specified sigma with invalid length: ' + str(len(sigma)))
        sigma = np.asarray(sigma, dtype=np.float64)
    else:
        sigma = np.full(dim, sigma, dtype=np.float64)
    # sigma is now a numpy array of length dim

    # check x0
    if isinstance(x0, (list, tuple, np.ndarray)):
        if len(x0) != dim:
            raise RuntimeError('specified x0 with invalid length: ' + str(len(x0)))
        x0 = np.asarray(x0, dtype=np.float64)
    else:
        x0 = np.full(dim, x0, dtype=np.float64)
    # x0 is now a numpy array of length dim

    # check step
    if isinstance(step, (list, tuple, np.ndarray)):
        if len(step) != dim:
            raise RuntimeError('specified step with invalid length: ' + str(len(step)))
        step = np.asarray(step, dtype=np.float64)
    else:
        step = np.full(dim, step, dtype=np.float64)
    # step is now a numpy array of length dim

    # check size
    if isinstance(size, (list, tuple, np.ndarray)):
        if len(size) != dim:
            raise RuntimeError('specified size with invalid length: ' + str(len(size)))
        size = tuple(size)
    elif dim != 1:
        raise RuntimeError('array size should be array-like of length: ' + str(dim))
    else:
        size = (size, )
    # size is now a tuple of length dim

    # compute weight that corresponds to specified max_dist
    max_dist_weight = exp(-max_dist**2/2)

    if method == 'optimized_convolution_S2':
        return _interpolate_opt_convol_S2(
            pts, val, sigma, x0, step, size, num_iter, max_dist_weight, resample,
            lambert_proj=lambert_proj, lambert_grid=lambert_grid, auto_proj=auto_proj)
        
    elif method == 'naive_S2':
        return _interpolate_naive_S2(pts, val, sigma, x0, step, size)
        
    else:
        raise RuntimeError("encountered invalid Barnes interpolation method: " + str(method))
    

# -----------------------------------------------------------------------------

def _interpolate_opt_convol_S2(
        pts, val, sigma, x0, step, size, num_iter, max_dist_weight, resample,
        lambert_proj=None, lambert_grid=None, auto_proj=True):
    """ 
    Implements the optimized convolution algorithm B for the unit sphere S^2.
    """
    # # the used Lambert projection
    # lambert_proj = get_lambert_proj()
    
    # # the *fixed* grid in Lambert coordinate space
    # lam_x0 = np.asarray([-32.0, -2.0])
    # lam_size = (int(44.0/step), int(64.0/step))
    
    # # map lonlat sample point coordinatess to Lambert coordinate space
    # lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)
    
    # # call ordinary 'optimized_convolution' algorithm
    # lam_field = interpolation._interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter)
    
    # if resample:
    #     return _resample(lam_field, lam_x0, x0, step, size, *lambert_proj)
    # else:
    #     return lam_field
    
    
    
    # split commented code above in two separately 'measurable' sub-routines
    # the convolution part taking place in Lambert space
    res1 = interpolate_opt_convol_S2_part1(
        pts, val, sigma, x0, step, size, num_iter, max_dist_weight,
        lambert_proj=lambert_proj, lambert_grid=lambert_grid, auto_proj=auto_proj)
    
    # the resampling part that performs back-projection from Lambert to lonlat space
    if resample:
        return interpolate_opt_convol_S2_part2(*res1)
    else:
        return res1[0]


def interpolate_opt_convol_S2_part1(
        pts, val, sigma, x0, step, size, num_iter, max_dist_weight,
        lambert_proj=None, lambert_grid=None, auto_proj=True):
    """ The convolution part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    if lambert_proj is None:
        if not auto_proj:
            raise RuntimeError('lambert_proj must be specified when auto_proj is False')
        lambert_proj = _infer_lambert_proj(x0, step, size)
    else:
        lambert_proj = tuple(lambert_proj)
        if len(lambert_proj) != 5:
            raise RuntimeError('lambert_proj must have length 5')

    if lambert_grid is None:
        if not auto_proj:
            raise RuntimeError('lambert_grid must be specified when auto_proj is False')
        margin_steps = interpolation.get_half_kernel_size_opt(float(np.max(sigma)), float(np.min(step)), num_iter) + 2
        lam_x0, lam_size = _infer_lambert_grid(x0, step, size, lambert_proj, margin_steps)
    else:
        lam_x0, lam_size = lambert_grid
        lam_x0 = np.asarray(lam_x0, dtype=np.float64)
        if len(lam_x0) != 2:
            raise RuntimeError('lambert_grid[0] must be length-2 array-like')
        if len(lam_size) != 2:
            raise RuntimeError('lambert_grid[1] must be length-2 tuple/list')
        lam_size = (int(lam_size[0]), int(lam_size[1]))
        if lam_size[0] < 2 or lam_size[1] < 2:
            raise RuntimeError('lambert_grid size must be >= (2,2)')
    
    # map lonlat sample point coordinates to Lambert coordinate space
    lam_pts = lambert_conformal.to_map(pts, pts.copy(), *lambert_proj)
    
    # call ordinary 'optimized_convolution' algorithm
    lam_field = interpolation._interpolate_opt_convol(lam_pts, val, sigma, lam_x0, step, lam_size, num_iter, max_dist_weight)

    return (lam_field, lam_x0, x0, step, size, lambert_proj)


@njit
def interpolate_opt_convol_S2_part2(lam_field, lam_x0, x0, step, size, lambert_proj):
    """ The back-projection part of _interpolate_opt_convol_S2(), allowing to measure split times. """
    return _resample(lam_field, lam_x0, x0, step, size, *lambert_proj)

    
def get_lambert_proj(center_lon=11.5, center_lat=34.5, lat1=42.5, lat2=65.5):
    """ Creates and returns Lambert projection constants. """
    return lambert_conformal.create_proj(center_lon, center_lat, lat1, lat2)


def _infer_lambert_proj(x0, step, size):
    """ Infers Lambert projection parameters from the requested lon-lat grid. """
    lon0 = float(x0[0])
    lon1 = float(x0[0] + (size[0]-1) * step[0])
    lat0 = float(x0[1])
    lat1 = float(x0[1] + (size[1]-1) * step[1])

    lon_min = min(lon0, lon1)
    lon_max = max(lon0, lon1)
    lat_min = min(lat0, lat1)
    lat_max = max(lat0, lat1)

    lon_span = lon_max - lon_min
    if lon_span >= 180.0:
        raise RuntimeError('optimized_convolution_S2 requires longitude span < 180 degrees')

    center_lon = 0.5 * (lon_min + lon_max)
    center_lat = 0.5 * (lat_min + lat_max)

    # Use standard parallels that are robust for mid-latitudes in each hemisphere.
    if lat_min >= 0.0:
        std1, std2 = 30.0, 60.0
    elif lat_max <= 0.0:
        std1, std2 = -30.0, -60.0
    else:
        raise RuntimeError('optimized_convolution_S2 does not support domains crossing the equator; split into hemispheres')

    return get_lambert_proj(center_lon, center_lat, std1, std2)


def _infer_lambert_grid(x0, step, size, lambert_proj, margin_steps):
    """ Infers Lambert grid origin and size from lon-lat grid corners and margins. """
    lon0 = float(x0[0])
    lon1 = float(x0[0] + (size[0]-1) * step[0])
    lat0 = float(x0[1])
    lat1 = float(x0[1] + (size[1]-1) * step[1])

    corners = np.asarray([
        [lon0, lat0],
        [lon0, lat1],
        [lon1, lat0],
        [lon1, lat1],
    ], dtype=np.float64)
    lam_corners = lambert_conformal.to_map(corners, corners.copy(), *lambert_proj)

    min_x = float(np.min(lam_corners[:, 0]))
    max_x = float(np.max(lam_corners[:, 0]))
    min_y = float(np.min(lam_corners[:, 1]))
    max_y = float(np.max(lam_corners[:, 1]))

    margin_x = margin_steps * float(step[0])
    margin_y = margin_steps * float(step[1])

    lam_x0 = np.asarray([min_x - margin_x, min_y - margin_y], dtype=np.float64)
    lam_size_x = int(np.ceil((max_x - min_x + 2.0*margin_x) / step[0])) + 2
    lam_size_y = int(np.ceil((max_y - min_y + 2.0*margin_y) / step[1])) + 2
    lam_size = (max(2, lam_size_x), max(2, lam_size_y))
    return lam_x0, lam_size

    
@njit
def _resample(lam_field, lam_x0, x0, step, size, center_lon, n, n_inv, F, rho0):
    """ Resamples the Lambert grid field to the specified lonlat grid. """
    # x-coordinate in lon-lat grid is constant over all grid lines
    geox = np.empty(size[0], dtype=np.float64)
    for i in range(size[0]):
        geox[i] = x0[0] + i*step[0]
        
    # memory for coordinates in Lambert space
    mapx = np.empty(size[0], dtype=np.float64)
    mapy = np.empty(size[0], dtype=np.float64)
    
    # memory for the corresponding Lambert grid indices 
    indx = np.empty(size[0], dtype=np.int32)
    indy = np.empty(size[0], dtype=np.int32)
    
    # memory for the resulting field in lonlat space
    rsize = size[::-1]
    res_field = np.empty(rsize, dtype=np.float32)
    
    # for each line in lonlat grid 
    for j in range(size[1]):
        # compute corresponding locations in Lambert space
        lambert_conformal.to_map2(geox, j*step[1] + x0[1], mapx, mapy, center_lon, n, n_inv, F, rho0)
        # compute corresponding Lambert grid indices
        mapx -= lam_x0[0]
        mapx /= step[0]
        mapy -= lam_x0[1]
        mapy /= step[1]
        # the corresponding 'i,j'-integer indices of the lower left grid point
        indx[:] = mapx.astype(np.int32)
        indy[:] = mapy.astype(np.int32)
        # and compute bilinear weights
        mapx -= indx    # contains now the weights
        mapy -= indy    # contains now the weights
        
        # compute bilinear interpolation of the 4 neighboring grid point values
        # and guard against out-of-bounds accesses at map edges.
        for i in range(size[0]):
            if indx[i] < 0 or indy[i] < 0 or indx[i] >= lam_field.shape[1]-1 or indy[i] >= lam_field.shape[0]-1:
                res_field[j,i] = np.nan
            else:
                res_field[j,i] = (1.0-mapy[i])*(1.0-mapx[i])*lam_field[indy[i],indx[i]] + \
                    mapy[i]*(1.0-mapx[i])*lam_field[indy[i]+1,indx[i]] + \
                    mapy[i]*mapx[i]*lam_field[indy[i]+1,indx[i]+1] + \
                    (1.0-mapy[i])*mapx[i]*lam_field[indy[i],indx[i]+1]
        
    return res_field
    

# -----------------------------------------------------------------------------

@njit
def _interpolate_naive_S2(pts, val, sigma, x0, step, size):
    """ Implements the naive Barnes interpolation algorithm A for the unit sphere S^2. """
    offset = interpolation._normalize_values(val)

    # the grid field to store the interpolated values - reverse grid dimensions
    rsize = size[::-1]
    grid_val = np.zeros(rsize, dtype=np.float64)

    scale = 2*sigma**2
    for j in range(size[1]):
        # compute y-coordinate of grid point
        yc = x0[1] + j*step[1]
        for i in range(size[0]):
            # compute x-coordinate of grid point
            xc = x0[0] + i*step[0]

            # use numpy to directly compute numerator and denominator of equ. (1)
            dist = _dist_S2(xc, yc, pts[:,0], pts[:,1])
            weight = np.exp(-dist*dist/scale[0])        # assuming scale is equal in x and y direction
            weighted_sum = np.dot(weight, val)
            weight_total = np.sum(weight)

            if weight_total > 0.0:
                grid_val[j,i] = weighted_sum / weight_total + offset
            else:
                grid_val[j,i] = np.nan

    return grid_val



RAD_PER_DEGREE = pi / 180.0


@njit
def _dist_S2(lon0, lat0, lon1, lat1):
    """ Computes spherical distance between the 2 specified points. Input and output in degrees. """
    lat0_rad = lat0 * RAD_PER_DEGREE
    lat1_rad = lat1 * RAD_PER_DEGREE
    arg = np.sin(lat0_rad)*np.sin(lat1_rad) + np.cos(lat0_rad)*np.cos(lat1_rad)*np.cos((lon1-lon0)*RAD_PER_DEGREE)
    arg[arg > 1.0] = 1.0
    return np.arccos(arg) / RAD_PER_DEGREE
