import numpy as np
import matplotlib.pyplot as plt

from geometrik import *
from metrics import curvature_mse
from test_utils import *


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
        k1 = equi_affine_curvature(X)
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


def test_invariance():

    np.random.seed(1)

    _eps = 1e-4
    itrs = 50

    def _coef_of_var(m):
        mu = np.mean(m, axis=0)
        sd = np.std(m, axis=0)
        return np.median(sd / np.maximum(mu, _eps))

    def _do_test(geom, tform_geom):
        K = np.zeros((itrs, len(X)))
        S = np.zeros((itrs, len(X)))
        for itr in range(itrs):
            XX = rand_transform(X, tform_geom)
            K[itr, :] = curvature(XX, geom)
            S[itr, :] = arclen(XX, geom)

        print(f'{geom.value} properties, under {tform_geom.value} transform:')
        print(" Curvature std-vs-avg: {:2.3f}".format(_coef_of_var(K)))
        print(" Arclength std-vs-avg: {:2.3f}".format(_coef_of_var(S)))

    X, _ = get_shape_points('sine')
    _do_test(geom=GEOMETRY.FULL_AFFINE, tform_geom=GEOMETRY.FULL_AFFINE)
    _do_test(geom=GEOMETRY.EQUI_AFFINE, tform_geom=GEOMETRY.FULL_AFFINE)
    _do_test(geom=GEOMETRY.EQUI_AFFINE, tform_geom=GEOMETRY.EQUI_AFFINE)
    _do_test(geom=GEOMETRY.EUCLIDEAN, tform_geom=GEOMETRY.EQUI_AFFINE)
    _do_test(geom=GEOMETRY.EUCLIDEAN, tform_geom=GEOMETRY.EUCLIDEAN)


def show_geometric_properties(shape):
    X, _ = get_shape_points(shape)
    _, axs = plt.subplots(nrows=2, ncols=3)
    for ix, geom in enumerate(GEOMETRIES):
        axs[0, ix].plot(arclen(X, geom), 'r')
        axs[1, ix].plot(curvature(X, geom), 'c')
        axs[0, ix].set_title(geom.value)
    axs[0, 0].set_ylabel('Arclen')
    axs[1, 0].set_ylabel('Curvature')
    plt.suptitle(shape)
    plt.show()


def test_metrics(shape=None):

    shape = 'ellipse' if shape is None else shape

    X, _ = get_shape_points(shape)

    print("Testing metric using shape=" + shape)

    def _do_test(metric_geom, tform_geom):
        # metric = make_randstable(curvature_mse, metric_geom, itrs=10)
        # mse = metric(X, rand_transform(X, tform_geom))
        mse = curvature_mse(X, rand_transform(X, tform_geom), metric_geom)
        assert (metric_geom.value == tform_geom.value) == (mse < 1e-2), \
            "MSE should be ~0 iff the transform geometry is the same as the metric geometry"
        print("{} transform, {} metric : {:2.2f}".format(
            tform_geom.value, metric_geom.value, mse))

    _do_test(metric_geom=GEOMETRY.EUCLIDEAN, tform_geom=GEOMETRY.EUCLIDEAN)
    _do_test(metric_geom=GEOMETRY.EUCLIDEAN, tform_geom=GEOMETRY.EQUI_AFFINE)
    _do_test(metric_geom=GEOMETRY.EUCLIDEAN, tform_geom=GEOMETRY.FULL_AFFINE)
    _do_test(metric_geom=GEOMETRY.EQUI_AFFINE, tform_geom=GEOMETRY.FULL_AFFINE)
    _do_test(metric_geom=GEOMETRY.EQUI_AFFINE, tform_geom=GEOMETRY.EQUI_AFFINE)
    _do_test(metric_geom=GEOMETRY.FULL_AFFINE, tform_geom=GEOMETRY.FULL_AFFINE)


if __name__ == "__main__":
    #show_geometric_properties('sine')
    #test_invariance()
    test_metrics()
