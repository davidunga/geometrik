from geometrik import *


def make_randstable(metric_fnc, geom: GEOMETRY, itrs=50):
    def fn(X1, X2):
        results = np.zeros(itrs)
        for itr in range(itrs):
            results[itr] = metric_fnc(X1, rand_transform(X2, geom), geom)
        return np.mean(results)
    return fn


def curvature_mse(X1, X2, geom: GEOMETRY):
    k_1 = curvature(uniform_resample(X1, geom)[0], geom)
    k_2 = curvature(uniform_resample(X2, geom)[0], geom)
    return np.mean((k_1 - k_2) ** 2)

