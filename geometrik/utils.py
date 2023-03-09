import numpy as np
from scipy.interpolate import interp1d
from geometrik.geometries import GEOMETRY


def simplify_curve(X, eps, rtol=None):
    """
    Ramer–Douglas–Peucker.
    Given a curve X, returns indices of points in X, such that
        the maximal distance between X and X[indices] is eps.
    """

    vec = X[-1] - X[0]
    norm = np.linalg.norm(vec)

    def _dist_to_line(pt):
        """ distance from a point to the line from X[0] to X[-1] """
        if norm == 0:
            return np.linalg.norm(X[0] - pt)
        return np.abs(np.linalg.norm(np.cross(vec, X[0] - pt))) / norm

    dists = [_dist_to_line(x) for x in X]
    if rtol is None:
        i = np.argmax(dists)
    else:
        i, i2 = np.argpartition(dists, -2)[-2:]
        if dists[i2] > dists[i]:
            i, i2 = i2, i
        if dists[i] > rtol * dists[i2]:
            i = i2
    if dists[i] > eps:
        return np.r_[simplify_curve(X[:i+1], eps, rtol)[:-1], i + simplify_curve(X[i:], eps, rtol)]
    return np.array([0, len(X) - 1])


def spatially_separated_points(X: np.ndarray, delta: float, force_last=True, momentum=True):
    """
    Get a subset of [X], such that the distance between two consecutive points is at least [delta].
    Args:
        X: N*D array of N D-dimensional points.
        delta: minimal separation.
        force_last: include last point, even if separation criterion is not met.
        momentum: demand that the separation keeps increasing. reduces noise.
    Returns:
        index array, ixs, such that |X[ixs[i]] - X[ixs[i+1]]| >= delta
    """

    delta2 = delta ** 2

    def _dist2(i_, j_):
        return ((X[i_] - X[j_]) ** 2).sum()

    bxs = np.zeros(len(X), bool)
    last_ix = 0
    bxs[0] = True
    for i in range(1, len(X)):
        if _dist2(last_ix, i) >= delta2 and (
                (not momentum) or
                (i == len(X) - 1) or
                _dist2(last_ix, i) < _dist2(last_ix, i + 1)):
            bxs[i] = True
            last_ix = i

    if force_last:
        bxs[last_ix] = False
        bxs[-1] = True
    return bxs


def randmat(geom=None, det=None, ortho=None, trns=False):
    """
    Random 2d square matrix
    :param det: If given, matrix will be scaled to have this determinant
    :param ortho: If True, matrix will be orthogonal
    """

    if (geom is not None) and not (det is None and ortho is None):
        raise ValueError("geom cannot be specified together with det or ortho")

    ortho = False if ortho is None else ortho

    if geom is not None:

        if geom.value == GEOMETRY.FULL_AFFINE.value:
            det_range = [.1, 10.0]  # constrain the determinant to avoid singularities
            det = (det_range[1] - det_range[0]) * np.random.rand() + det_range[0]
        elif geom.value == GEOMETRY.EQUI_AFFINE.value:
            det = 1
        elif geom.value == GEOMETRY.EUCLIDEAN.value:
            det = 1
            ortho = True
        else:
            raise ValueError("Unknown geometry")

    m = np.random.rand(2, 2)
    if ortho:
        m[:, 1] = [m[1, 0], -m[0, 0]]
        assert np.abs(np.dot(m[:, 0], m[:, 1])) < 1e-6
    if det is not None:
        m *= np.sqrt(np.abs(det / np.linalg.det(m)))
        if np.sign(np.linalg.det(m)) != np.sign(det):
            m = np.fliplr(m)
        assert np.abs(np.linalg.det(m) - det) < 1e-6
    assert np.linalg.matrix_rank(m) == 2

    if trns:
        m = np.c_[m, np.random.randn(2, 1)]

    return m


def rand_transform(X: np.ndarray, geom: GEOMETRY):
    """
    Randomly transform curve in a manner that maintains invariance under given geometry
    :param X: Curve to transform (2d ndarray)
    :param geom: Geometry for invariance
    :return: Transformed curve (same size as input)
    """
    return np.dot(randmat(geom=geom), X.T).T


def extrap_boundaries(y, x=None, b=3):
    """
    Replace values in array y, with an extrapolation
    :param b: boundary size
    """
    x = list(range(len(y))) if x is None else x
    return interp1d(x[b:-b], y[b:-b], fill_value='extrapolate', kind='linear')(x)


def derivative(X, t, n=1):
    """
    n-th order derivatives of X wrt t
    :param X: 2d array, NxD, N = points, D = dim
    :param t: 1d array, parameter to differentiate by.
    :param n: order of derivative
    :return: drvs: list of length n, drvs[i] contains the (i+1)-th order derivative of X
    """
    X = np.expand_dims(X, axis=1) if X.ndim == 1 else X
    drvs = []
    for k in range(n):
        X = np.copy(X)
        for j in range(X.shape[1]):
            X[:, j] = np.gradient(X[:, j], t, edge_order=1)
        drvs.append(X.squeeze())
    return drvs


def winding_angle(X: np.ndarray):
    """
    :param X: Planar curve given as Nx2 ndarray
    :return: thetas - 1d nparray same length as X. thetas[i] is the winding
        angle at X[i]
    """
    dX = np.diff(X, axis=0)
    thetas = np.unwrap(np.arctan2(dX[:, 1], dX[:, 0]))
    thetas = np.concatenate([thetas, [2 * thetas[-1] - thetas[-2]]])
    return thetas


def inflection_points(X: np.ndarray):
    """ Index of inflection points in curve """
    ixs = np.nonzero(np.diff(np.sign(np.diff(winding_angle(X)))))[0]
    if len(ixs) > 0:
        ixs += 2
    return ixs


def is_convex(X: np.ndarray):
    """
    Is planar curve angular-convex?
    :param X: Planar curve given as Nx2 ndarray
    """
    return len(inflection_points(X)) == 0


def angdiff(a, b):
    """ Absolute difference between angles (in radians) """
    return np.pi - abs(np.mod(abs(a - b), 2 * np.pi) - np.pi)


def fourier(x, t):
    """
    FFT of x(t)
    :param x: 1d array (can be complex)
    :param t: array same size as x, such that t[i] is the sample time of x[i]
    :return:
        F, frq - 1d array, half the length of x.
        F[i] is the fourier coeff for frequency frq[i]
    """
    F = np.fft.fft(x)
    n = int(np.ceil(.5 * len(t)) + 1)
    w = 2 * np.pi / (t[-1] - t[0])
    frq = np.arange(0, n - 1) * w
    F = F[:n]
    return F, frq


def calc_affine_tform(X: np.ndarray, Y: np.ndarray):
    """
    Find affine transform which brings X to Y, in the least-squares sense.
    i.e., find a matrix A, s.t. the error |Y-AX| is minimized.
    :param X: Planar curve given as Nx2 ndarray
    :param Y: Planar curve given as Nx2 ndarray
    :return: A - 2x3 matrix, the top two rows of the transformation matrix,
        the full matrix is given by: [A,[0,0,1]]
    """
    pad = np.ones(len(X))
    A = np.c_[Y, pad].T @ np.linalg.pinv(np.c_[X, pad].T)
    return A[:2]


def apply_affine_tform(A: np.ndarray, X: np.ndarray):
    """
    Transform curve X using affine matrix A
    :param A: 2x3 affine matrix
    :param X: Planar curve given as Nx2 ndarray
    :return: Curve, same size as X
    """
    return (A[:, :2] @ X.T).T + A[:, 2]


def procrustes_metric(X: np.ndarray, Y: np.ndarray):
    """
    Distance between curves X & Y
    :param X: Planar curve given as Nx2 ndarray
    :param Y: Planar curve given as Nx2 ndarray
    :return: scalar. the distance between X & Y
    """
    return np.sum((Y - X) ** 2) / np.sum((X - X.mean(axis=0)) ** 2)
