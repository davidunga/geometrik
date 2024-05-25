import numpy as np
from scipy.interpolate import interp1d
from geometrik.spcurves import NDSpline, NumericCurve
from scipy.ndimage import gaussian_filter1d


def numeric_arclen(X):
    return np.r_[[0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]


def uniform_resample(X):
    s = numeric_arclen(X)
    u = np.linspace(s[0], s[-1], len(s))
    Xu = interp1d(s, X, axis=0, kind="cubic")(u)
    return Xu, u, s


def estimate_sampling_scale(X):
    return .001 * np.median(np.abs(np.diff(X, axis=0)))


def make_ndspline(X: np.ndarray, t: np.ndarray = None, dx: float = None,
                  k: int = 5, stol: float = .0, default_t: np.ndarray = None, smooth_sig: float = .0) -> NDSpline:
    """
    Args:
        X: Trajectory points
        t: Time vec
        dx: Minimal meaningful distance (Spatial resolution)
        dst_t: Destination time vec. default = uniform sampling of t
        k: spline order
        stol: smoothing tolerance factor, relative to dx. spline smoothing parameter (s) = stol * dx * len(X)
    """

    Xss, tss, dx, ss_ixs = process_sampled_curve(X=X, t=t, dx=dx, smooth_sig=smooth_sig)
    return NDSpline(Xss, t=tss, k=k, w=None, s=stol * dx * len(X), default_t=default_t)


def make_numeric_curve(X: np.ndarray, t: np.ndarray, dx: float, dst_t: np.ndarray = None):
    X, t = process_sampled_curve(X, t, dx)[:2]
    return NumericCurve(X, t, dst_t=dst_t)


def process_sampled_curve(X: np.ndarray, t: np.ndarray, dx: float = None, smooth_sig: float = .0):
    """
        Args:
            X: Trajectory points
            t: Time vec
            dx: Minimal meaningful distance (Spatial resolution)
            smooth_sig: Sigma for Gaussian smoothing
        Returns:
            X: Processed trajectory, same size as input
        """

    if dx is None:
        dx = estimate_sampling_scale(X)
    if t is None:
        t = np.arange(len(X)) / len(X)
    assert np.all(np.diff(t) > 0)
    ss_ixs = spatially_separated_points(X, delta=dx, momentum=True)
    Xss = X[ss_ixs]
    tss = t[ss_ixs]
    if smooth_sig > 0:
        Xss = gaussian_filter1d(Xss, sigma=smooth_sig / dx, axis=0, mode='mirror')
    return Xss, tss, dx, ss_ixs


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
