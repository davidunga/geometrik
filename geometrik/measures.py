"""
Curves measures, i.e., curvature and arclength, under different geometries
"""

import numpy as np
from geometrik.geometries import GEOMETRY, GEOMETRIES
from geometrik.utils import derivative, extrap_boundaries, winding_angle
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# =========================================================


def _check_curve(X: np.ndarray, geom: GEOMETRY = None):
    assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 2,\
        "Curve must be Nx2 ndarray"


# =========================================================

def arclen(X: np.ndarray, geom: GEOMETRY):
    """
    Cumulative arc length of curve, in a given geometry.
    :param X: Planar curve as Nx2 ndarray
    :param geom: Geometry to use.
    :return: s - 1d array, s[i] is the arclength from X[0] to X[i].
        s[-1] is the total arclength.
        s[0] is always zero.
    """
    assert geom in GEOMETRIES
    return {
        GEOMETRY.EUCLIDEAN: euclidean_arclen,
        GEOMETRY.EQUI_AFFINE: equi_affine_arclen,
        GEOMETRY.FULL_AFFINE: full_affine_arclen
    }[geom](X)


def curvature(X: np.ndarray, geom: GEOMETRY):
    """
    Curvature at each point along path, measured in a given geometry.
    :param X: Planar curve as Nx2 ndarray
    :param geom: Geometry to use.
    :return: k: 1d array of length N. k[i] is the curvature at X[i], (nan if undefined).
    """
    assert geom in GEOMETRIES
    return {
        GEOMETRY.EUCLIDEAN: euclidean_curvature,
        GEOMETRY.EQUI_AFFINE: equi_affine_curvature,
        GEOMETRY.FULL_AFFINE: full_affine_curvature
    }[geom](X)


def uniform_resample(X: np.ndarray, geom: GEOMETRY):
    """
    Resample path uniformly, according to a given geometry.
    :param X: Curve as Nx2 numpy array.
    :param geom: Geometry to go by
    :return: resampled path, same size as input
    """
    _check_curve(X, geom)
    s_current = arclen(X, geom)
    s_uniform = np.linspace(0, s_current[-1], len(X))
    X = np.copy(X)
    for j in range(X.shape[1]):
        X[:, j] = interp1d(s_current, X[:, j], kind='cubic')(s_uniform)
    return X, s_uniform


def arclens_and_curvatures(X: np.ndarray):
    """
    Arc length and curvature of curve in all 3 geometries
    :param X: Planar curve as Nx2 numpy array
    :return:
        s0 = 1d numpy array. s0[i] is the full-affine arc length from X[0] to X[i].
        s1 = 1d numpy array. s1[i] is the equi-affine arc length from X[0] to X[i].
        s2 = 1d numpy array. s2[i] is the euclidean arc length from X[0] to X[i].
        k0 = 1d numpy array. k0[i] is the full-affine curvature around point X[i].
        k1 = 1d numpy array. k1[i] is the equi-affine curvature around point X[i].
        k2 = 1d numpy array. k2[i] is the euclidean curvature around point X[i].
    """
    s2 = euclidean_arclen(X)
    k2 = euclidean_curvature(X)
    s1 = equi_affine_arclen(X)
    k1 = equi_affine_curvature(X, s1)
    s0 = full_affine_arclen(X, k1, s1)
    k0 = full_affine_curvature(X, k1, s1)
    return s0, s1, s2, k0, k1, k2


# =========================================================


def equi_affine_arclen(X: np.ndarray):
    """
    Cumulative equi-affine arc length of curve
    :param X: Planar curve as Nx2 numpy array
    :return: s1 - 1d array, s1[i] is the arclength from X[0] to X[i].
        s1[-1] is the total arclength.
        s1[0] is always zero.
    """
    _check_curve(X)
    t = np.linspace(0, 1, len(X))
    dX1, dX2 = derivative(X, t, n=2)
    dets = dX1[:, 0] * dX2[:, 1] - dX1[:, 1] * dX2[:, 0]
    s1 = np.concatenate([[0], cumulative_trapezoid(np.abs(dets) ** (1 / 3), t)])
    return s1


def equi_affine_curvature(X: np.ndarray, s1: np.ndarray = None):
    """
    Pointwise equi-affine curvature
    :param X: Planar curve as Nx2 numpy array
    :param s1: 1d ndarray, equi-affine arclength (to avoid re-computing)
    :return: k1 - 1d ndarray, k1[i] is the curvature around point X[i]
    """
    _check_curve(X)
    s1 = equi_affine_arclen(X) if s1 is None else s1
    dX1, dX2, dX3 = derivative(X, s1, n=3)
    k1 = dX2[:, 0] * dX3[:, 1] - dX2[:, 1] * dX3[:, 0]
    k1 = extrap_boundaries(k1, s1, b=5)
    return k1


def full_affine_arclen(X: np.ndarray, k1: np.ndarray = None, s1: np.ndarray = None):
    """
    Cumulative full-affine arc length of curve
    :param X: Planar curve as Nx2 numpy array
    :return: s0 - 1d array, s0[i] is the arclength from X[0] to X[i].
        s0[-1] is the total arclength.
        s0[0] is always zero.
    """
    _check_curve(X)
    if s1 is None:
        s1 = equi_affine_arclen(X)
    if k1 is None:
        k1 = equi_affine_curvature(X, s1)
    s0 = np.concatenate([[0], cumulative_trapezoid(np.sqrt(np.abs(k1)), s1)])
    return s0


def full_affine_curvature(X: np.ndarray, k1: np.ndarray = None, s1: np.ndarray = None):
    """
    Pointwise full-affine curvature
    :param X: Planar curve as Nx2 numpy array
    :param k1: 1d ndarray, equi-affine curvature (to avoid re-computing)
    :param s1: 1d ndarray, equi-affine arclength (to avoid re-computing)
    :return: k0 - 1d ndarray, k0[i] is the curvature around point X[i]
    """
    _check_curve(X)
    _eps = 1e-4
    if s1 is None:
        s1 = equi_affine_arclen(X)
    if k1 is None:
        k1 = equi_affine_curvature(X, s1)
    k1_abs = np.abs(k1)
    k1_drv = derivative(k1, s1)[0]

    valid_ixs = k1_abs > _eps
    nonzero_ixs = np.abs(k1_drv) > _eps

    k0 = np.zeros_like(k1)
    ii = valid_ixs & nonzero_ixs
    k0[ii] = (k1_abs[ii] ** -1.5) * k1_drv[ii]
    k0[~valid_ixs] = np.nan
    return k0


def euclidean_arclen(X: np.ndarray):
    """
    Cumulative euclidean arc length of a curve
    :param X: Planar curve as Nx2 ndarray
    :return: s2 - 1d array, s2[i] is the arclength from X[0] to X[i].
        s2[-1] is the total arclength.
        s2[0] is always zero.
    """
    _check_curve(X, GEOMETRY.EUCLIDEAN)
    s2 = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(np.diff(X, axis=0) ** 2, axis=1)))])
    return s2


def euclidean_curvature(X: np.ndarray):
    """
    Pointwise Euclidean curvature
    :param X: Planar curve as Nx2 ndarray
    :return: k2 - 1d ndarray, k2[i] is the curvature around point X[i]
    """
    _check_curve(X, GEOMETRY.EUCLIDEAN)
    s = euclidean_arclen(X)
    t = winding_angle(X)
    k2 = derivative(s, t)[0]
    return k2

