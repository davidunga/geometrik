
import numpy as np
from geometrik.utils import rand_transform
from geometrik.geometries import GEOMETRY, GEOMETRIES, convention2geom
from collections import defaultdict
from test_utils import get_shape_points
from geometrik.spcurves import NDSpline
from geometrik.invariants import geometric_invariants
from geometrik.spcurve_factory import make_ndspline, make_numeric_curve, uniform_resample
import matplotlib.pyplot as plt


def _test():
    """
    Test that invariants are indeed invariant exactly under the correct geometries.
    For example, the equi-affine arc-length should remain unchanged under euclidean and equi-affine
    transformations, and should change under full affine transformations.
    """

    itrs = 50
    np.random.seed(0)
    only_arclens = True

    def _coef_of_var(m):
        mu = np.maximum(np.mean(np.abs(m), axis=0), 1e-6)
        return np.median(np.std(np.abs(m), axis=0) / mu)

    results = []

    def _process_result(shape, msg, kind, success):
        results.append({"shape": shape, "msg": msg, "kind": kind, "success": success})
        print(msg)

    for shape in ('ellipse', 'parabola', 'hyperbola', 'nice_sine'):

        X, t, analytic_invars = get_shape_points(shape, tnoise=0.0001)

        #plt.plot(X[:,0],X[:,1],'.k')
        #plt.show()

        print("Testing on " + shape.upper() + ":")

        for tform_geom in GEOMETRIES:

            invars = defaultdict(list)

            for itr in range(itrs):
                XX = rand_transform(X, tform_geom)
                #spl = NDSpline(XX, k=5)

                #XX_uniform, s_uniform, s_current = uniform_resample(XX)
                #t = np.arange(len(X)) / len(X)

                spl = make_ndspline(XX, k=5, stol=1e-6, smooth_sig=.0005)
                for k, v in geometric_invariants(spl).items():
                    if only_arclens and not k.startswith('s'):
                        continue
                    invars[k].append(v)

            for k, v in invars.items():
                geom = convention2geom(k[-1])
                #if geom.value > tform_geom.value:
                #    continue

                v = np.stack(v, axis=0)

                if k in analytic_invars and geom == tform_geom:

                    if analytic_invars[k] == "UNDEF":
                        continue

                    avg = np.mean(v)
                    err = np.abs(avg - analytic_invars[k]) / max(abs(analytic_invars[k]), 1e-3)
                    is_same = err < .01

                    msg = f"{k} - analytic={analytic_invars[k]:2.2f} computed={avg:2.2f} err={err:2.2f}"
                    msg = f"{msg:<50s} - {'(NOT SAME)' if not is_same else '(same)'}"

                    _process_result(shape, msg, "match", is_same)

                    if analytic_invars[k] == 0:
                        cv = np.std(v)

                else:
                    cv = _coef_of_var(v)

                #if geom.value == GEOMETRY.FULL_AFFINE.value and tform_geom.value == GEOMETRY.FULL_AFFINE.value:
                #    plt.plot(v.T)
                #    plt.show()

                var_level = "mid"
                is_fail = False
                should_by_invar = geom.value >= tform_geom.value
                if cv < .1:
                    var_level = "low"
                    is_fail = not should_by_invar
                elif cv > 1:
                    var_level = "high"
                    is_fail = should_by_invar

                msg = f"{k} under {tform_geom.name} - " + var_level.upper() + " variance."
                msg = f"{msg:<40s} cv={cv:2.3f} avg={np.mean(v):2.1f} std={np.std(v, axis=0).mean():2.1f}"
                msg = f"{msg:<60s} - {'FAILED' if is_fail else ('success' if var_level != 'mid' else '(undetermined)')}"

                _process_result(shape, msg, "variance", not is_fail)

    is_success = [res["success"] for res in results]
    print(f"\n --> Score: {np.mean(is_success):.2%} ({np.sum(is_success)}/{len(is_success)})")
    if not np.all(is_success):
        print(" Fails recap: ")
        for res in results:
            if not res["success"]:
                print(f"{res['shape'].upper():<10s} - " + res['msg'])


if __name__ == "__main__":
    _test()
