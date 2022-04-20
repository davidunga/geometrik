import matplotlib.pyplot as plt

from geometrik import measures, metrics, utils
from geometrik.geometries import GEOMETRY, GEOMETRIES
from test_utils import get_shape_points
import numpy as np
from itertools import combinations_with_replacement


def test_equi_affine():

    # shapes with constant equi affine curvature:
    b = 2
    k1_per_shape = {
        'parabola': 0,
        'ellipse': b ** -2,
        'hyperbola': -b ** -2,
    }

    _, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

    for shape in k1_per_shape:
        X, _ = get_shape_points(shape, b=b)
        k1 = measures.equi_affine_curvature(X)
        err = np.max(np.abs(k1 - k1_per_shape[shape]))
        print(shape.upper())
        print("Expected: {:2.2f}, Got: {:2.2f}, Error={:2.2f}".format(
            k1_per_shape[shape], np.mean(k1), err))

        ax1.cla()
        ax2.cla()
        ax1.plot(X[:, 0], X[:, 1], 'r')
        ax2.plot(k1, 'r-')
        ax2.plot([0, len(k1) - 1], k1_per_shape[shape] * np.array([1, 1]), 'k')
        plt.waitforbuttonpress()

    plt.show()


def test_measures():

    np.random.seed(1)

    _eps = 1e-4
    itrs = 50

    def _coef_of_var(m):
        mu = np.mean(m, axis=0)
        sd = np.std(m, axis=0)
        return np.median(sd / np.maximum(mu, _eps))

    def _do_test(geom, tform_geom):
        assert geom.value <= tform_geom.value
        K = np.zeros((itrs, len(X)))
        S = np.zeros((itrs, len(X)))
        for itr in range(itrs):
            XX = utils.rand_transform(X, tform_geom)
            K[itr, :] = measures.curvature(XX, geom)
            S[itr, :] = measures.arclen(XX, geom)

        print(f'{geom.name} properties, under {tform_geom.name} transform:', end=" ")
        cv_k = _coef_of_var(K)
        cv_s = _coef_of_var(S)
        print("Std-vs-Avg: Curvature={:2.3f} Arclength={:2.3f}".format(cv_k, cv_s), end=" ")

        return (geom == tform_geom) == (max(cv_k, cv_s) < 1e-2)

    print("-- Invariance test:")
    print("Testing that G-curvature and G-arclength of a curve remain unchanged under G-transformation.")

    X, _ = get_shape_points('ellipse')
    results = []
    for (geom, tform_geom) in combinations_with_replacement(GEOMETRIES, 2):
        results.append(_do_test(geom=geom, tform_geom=tform_geom))
        print("PASSED" if results[-1] else "FAILED")

    print("Test " + ("PASSED" if all(results) else "FAILED"))


def show_geometric_properties(shape):
    X, _ = get_shape_points(shape)
    _, axs = plt.subplots(nrows=2, ncols=3)
    for ix, geom in enumerate(GEOMETRIES):
        axs[0, ix].plot(measures.arclen(X, geom), 'r')
        axs[1, ix].plot(measures.curvature(X, geom), 'c')
        axs[0, ix].set_title(geom.name)
    axs[0, 0].set_ylabel('Arclen')
    axs[1, 0].set_ylabel('Curvature')
    plt.suptitle(shape)
    plt.show()


def test_metrics(shape=None):

    shape = 'ellipse' if shape is None else shape

    X, _ = get_shape_points(shape)

    print("--- Metrics test:")
    print("Testing that G-distance between two g-similar curves is non-zero if and only if g != G")
    print("Testing metric using shape=" + shape)

    def _do_test(metric_geom, tform_geom):
        assert metric_geom.value <= tform_geom.value

        # metric = make_randstable(curvature_mse, metric_geom, itrs=10)
        # mse = metric(X, rand_transform(X, tform_geom))
        mse = metrics.curvprofile_pdist([X, utils.rand_transform(X, tform_geom)], metric_geom)
        print("Transform={:12s} Metric={:12s} MSE={:10.2g}".format(
            tform_geom.name.upper(), metric_geom.name.upper(), mse), end=" ")
        return (metric_geom == tform_geom) == (mse < 1e-2)

    results = []
    for (metric_geom, tform_geom) in combinations_with_replacement(GEOMETRIES, 2):
        results.append(_do_test(metric_geom=metric_geom, tform_geom=tform_geom))
        print("PASSED" if results[-1] else "FAILED")
    print("Test " + ("PASSED" if all(results) else "FAILED"))


if __name__ == "__main__":
    test_measures()
    test_metrics()
