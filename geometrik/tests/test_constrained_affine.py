from test_utils import get_shape_points
import matplotlib.pyplot as plt
from geometrik import utils
from geometrik.geometries import GEOMETRY, GEOMETRIES
import numpy as np
from geometrik.constrained_affine import find_transformation, TFORM

TFORMS = [t for t in TFORM]


def test(plot=True):

    print("--- Testing constrained affine curve matching")

    X, _ = get_shape_points('sine')
    success = []

    if plot:
        _, axs = plt.subplots(ncols=len(TFORMS))

    for tform_ix, tform in enumerate(TFORMS):

        geom = {
            TFORM.FULL_AFINE: GEOMETRY.FULL_AFFINE,
            TFORM.ORIENTATED_AFFINE: GEOMETRY.FULL_AFFINE,
            TFORM.EQUI_AFFINE: GEOMETRY.EQUI_AFFINE,
            TFORM.EUCLIDEAN: GEOMETRY.EUCLIDEAN
        }[tform]

        A_gt = utils.randmat(geom=geom, trns=True)
        if tform == TFORM.ORIENTATED_AFFINE:
            A_gt[:2, :2] *= -1

        Y = (A_gt[:2, :2] @ X.T + A_gt[:, 2:]).T

        A_est, Xt, opt_result = find_transformation(X, Y, tform=tform)
        success.append(np.isclose(A_est.flatten(), A_gt.flatten(), rtol=1.e-2, atol=1.e-2).all())
        print(tform.name + " : " + ("PASSED" if success[-1] else "FAILED"))

        if plot:
            Xt = (A_est[:2, :2] @ X.T).T + A_est[:, 2]
            for XX, marker in [(Y, 'ro'), (X, 'co'), (Xt, 'c-')]:
                axs[tform_ix].plot(XX[:, 0], XX[:, 1], marker)
                axs[tform_ix].plot(XX[0, 0], XX[0, 1], 's' + marker[0])
            axs[tform_ix].set_title(tform.name)

    print("Test " + ("PASSED" if all(success) else "FAILED"))

    if plot:
        plt.legend(['Y', 'X', 'Xt'])
        plt.show()


if __name__ == "__main__":
    test()


