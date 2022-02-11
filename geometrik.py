import numpy as np
from enum import Enum
from utils import derivative, extrap_boundaries, randmat
from scipy.integrate import cumtrapz

try:
    import matplotlib.pyplot as plt
except:
    pass


# =========================================================


class GEOMETRY(Enum):
    FULL_AFFINE = 'full_affine'
    EQUI_AFFINE = 'equi_affine'
    EUCLIDEAN = 'euclidean'


GEOMETRIES = [e for e in GEOMETRY]


def arclen(X, geom: GEOMETRY):
    """
    Cumulative arc length of path, in a given geometry,
    :param X: path, NxD, N = number of points, D = dimension
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


def curvature(X, geom: GEOMETRY):
    """
    Curvature at each point along path, measured in a given geometry.
    :param X: path, NxD, N = number of points, D = dimension.
    :param geom: Geometry to use.
    :return: k: 1d array of length N. k[i] is the curvature at X[i], (nan if undefined).
    """

    assert geom in GEOMETRIES
    return {
        GEOMETRY.EUCLIDEAN: euclidean_curvature,
        GEOMETRY.EQUI_AFFINE: equi_affine_curvature,
        GEOMETRY.FULL_AFFINE: full_affine_curvature
    }[geom](X)


def uniform_resample(X, geom: GEOMETRY):
    """
    Resample path uniformly
    :param X: path, NxD, N = number of points, D = dimension
    :param geom: Geometry to go by
    :return: resampled path, same size as input
    """
    s_current = arclen(X, geom)
    s_uniform = np.linspace(0, s_current[-1], len(X))
    X = np.copy(X)
    for j in range(X.shape[1]):
        X[:, j] = np.interp(s_uniform, s_current, X[:, j])
    return X, s_uniform


def arclens_and_curvatures(X):
    s2 = euclidean_arclen(X)
    k2 = euclidean_curvature(X)
    s1 = equi_affine_arclen(X)
    k1 = equi_affine_curvature(X, s1)
    s0 = full_affine_arclen(X, k1, s1)
    k0 = full_affine_curvature(X, k1, s1)
    return s0, s1, s2, k0, k1, k2


# =========================================================


def equi_affine_arclen(X):
    t = np.linspace(0, 1, len(X))
    dX1, dX2 = derivative(X, t, n=2)
    dets = dX1[:, 0] * dX2[:, 1] - dX1[:, 1] * dX2[:, 0]
    s1 = np.concatenate([[0], cumtrapz(np.abs(dets) ** (1 / 3), t)])
    return s1


def equi_affine_curvature(X, s1=None):
    s1 = equi_affine_arclen(X) if s1 is None else s1
    dX1, dX2, dX3 = derivative(X, s1, n=3)
    k1 = dX2[:, 0] * dX3[:, 1] - dX2[:, 1] * dX3[:, 0]
    k1 = extrap_boundaries(k1, s1, b=5)
    return k1


def full_affine_arclen(X, k1=None, s1=None):
    if s1 is None:
        s1 = equi_affine_arclen(X)
    if k1 is None:
        k1 = equi_affine_curvature(X, s1)
    s0 = np.concatenate([[0], cumtrapz(np.sqrt(np.abs(k1)), s1)])
    return s0


def full_affine_curvature(X, k1=None, s1=None):
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


def euclidean_arclen(X):
    s2 = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(np.diff(X, axis=0) ** 2, axis=1)))])
    return s2


def euclidean_curvature(X):
    assert len(X) > 2, "At least 3 points are required"
    assert X.shape[1] > 1, "Curvature is undefined for dim < 2"
    a = np.linalg.norm(X[1:-1] - X[:-2], axis=1)
    b = np.linalg.norm(X[2:] - X[1:-1], axis=1)
    c = np.linalg.norm(X[2:] - X[:-2], axis=1)
    k2 = np.zeros(len(X))
    k2[1:-1] = a * b * c / np.sqrt((a+b+c) * (b+c-a) * (c+a-b) * (a+b-c))
    k2 = extrap_boundaries(k2, b=1)
    return k2


def rand_transform(X, geom: GEOMETRY):
    """
    Randomly transform curve in a manner that maintains invariance under given geometry
    :param X: Curve to transform (2d ndarry)
    :param geom: Geometry for invariance
    :return: Transformed curve (same size as input)
    """
    if geom == GEOMETRY.FULL_AFFINE:
        det_range = [.2, 10.0]  # constrain the determinant to avoid singularities
        m = randmat(det=(det_range[1] - det_range[0]) * np.random.rand() + det_range[0])
    elif geom == GEOMETRY.EQUI_AFFINE:
        m = randmat(det=1)
    elif geom == GEOMETRY.EUCLIDEAN:
        m = randmat(det=1, ortho=True)
    else:
        raise ValueError("Unknown geometry")
    XX = np.dot(m, X.T).T
    return XX


if __name__ == "__main__":
    pass

