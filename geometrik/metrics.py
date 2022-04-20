"""
Metric of curve similarities under different representations & geometries
"""

from geometrik import measures
from geometrik.geometries import GEOMETRY
from geometrik import utils
from scipy.spatial.distance import pdist
import numpy as np


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
            results[itr] = metric_fnc(X1, utils.rand_transform(X2, geom), geom)
        return np.mean(results)
    return fn


def curvprofile_pdist(Xs, geom: GEOMETRY, oriented=False):
    """
    MSE between G-curvature profiles
    :param Xs: Array of (same-size) planar curvatures, each as Nx2 ndarray
    :param geom: Geometry to work under
    :param oriented: flag- if False (default), metric ignores curve orientation
    :param dist_metric: distance metric for comparing curvature profiles
    :return: mse of G-curvatures
    """

    def _canberra_norm(u, v):
        return np.nanmax(np.abs(u - v)) / (np.nanmax(np.abs(v)) + np.nanmax(np.abs(u)))

    ks = np.zeros((len(Xs), len(Xs[0])))
    for i in range(len(Xs)):
        ks[i] = measures.curvature(measures.uniform_resample(Xs[i], geom)[0], geom)
    if not oriented:
        ks = np.abs(ks)
    result = pdist(ks, metric=_canberra_norm)
    return result.squeeze()
