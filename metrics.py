"""
Metric of curve similarities under different representations & geometries
"""

from measures import *


def _make_randstable(metric_fnc, geom: GEOMETRY, itrs=50):
    """
    Wrapper for stabilizing metrics by re-computing it under random
    transformations which the metric is agnostic to.
    :param metric_fnc: metric to stabilize
    :param geom: geometry to work in
    :param itrs: number of random iterations
    :return: stabilize metric function
    """
    def fn(X1, X2):
        results = np.zeros(itrs)
        for itr in range(itrs):
            results[itr] = metric_fnc(X1, rand_transform(X2, geom), geom)
        return np.mean(results)
    return fn


def curvature_mse(X1: np.ndarray, X2: np.ndarray, geom: GEOMETRY, oriented=False, normalize=True):
    """
    MSE between G-curvature profiles
    :param X1: Planar curvature as Nx2 ndarray
    :param X2: Planar curvature as Nx2 ndarray
    :param geom: Geometry to work under
    :param oriented: flag- if False (default), metric ignores curve orientation
    :param normalize: flag- normalize by mean squared curvature
    :return: mse of G-curvatures
    """
    k_1 = curvature(uniform_resample(X1, geom)[0], geom)
    k_2 = curvature(uniform_resample(X2, geom)[0], geom)
    if not oriented:
        k_1 = np.abs(k_1)
        k_2 = np.abs(k_2)
    scale = 1 if not normalize else 1 / np.mean(k_1 ** 2 + k_2 ** 2)
    return np.mean((k_1 - k_2) ** 2) * scale
