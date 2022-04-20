import numpy as np
import scipy.optimize as opt
from geometrik import utils
from enum import Enum, auto
from geometrik.geometries import GEOMETRY


class TFORM(Enum):
    """
    Transformations of the form Ax + t, where A is subject to constraints
    """
    FULL_AFINE = auto()             # unconstrained
    ORIENTATED_AFFINE = auto()      # orientation preserving (det > 0)
    EQUI_AFFINE = auto()            # area and orientation (det = 1)
    EUCLIDEAN = auto()              # rigid (transpose = inverse)


def find_transformation(X, Y, tform: TFORM, maxiter=500, tol=1e-6):
    """
    For two curves X & Y, with corresponding points (X[i] should be matched with Y[i]),
    find a constrained affine transformation which brings X to Y, in the least sqaures sense.
    i.e., Find a matrix A such that |Y - AX| is minimized, where A is subject to constraints.
    :param X: Planar curve as 2d np array
    :param Y: Planar curve as 2d np array, same length as X
    :param tform: enum. which transformation should be used
    :param maxiter: max number of optimizer iterations
    :param tol: early stopping threshold in terms of MSE(Y, AX)
    :return:
        A - found transformation
        opt_result - optimization result
    """

    if tform == TFORM.FULL_AFINE:
        # unconstrained
        constraints = []
        # initial guess is the analytic solution
        A0 = utils.calc_affine_tform(X, Y)

    elif tform == TFORM.ORIENTATED_AFFINE:

        # constraint: det > 0
        constraints = opt.NonlinearConstraint(lambda Ap: np.linalg.det(AffineParams.R(Ap)), lb=0, ub=np.inf)

        # initial guess: analytic solution corrected for det > 0
        A0 = utils.calc_affine_tform(X, Y)
        if np.linalg.det(A0[:2, :2]) < 0:
            A0[:2, :2] *= 1

    elif tform == TFORM.EQUI_AFFINE:

        # constraint: det = 1
        constraints = opt.NonlinearConstraint(lambda Ap: np.linalg.det(AffineParams.R(Ap)), lb=1, ub=1)

        # initial guess: random matrix with det = 1
        A0 = utils.randmat(geom=GEOMETRY.EQUI_AFFINE, trns=True)
        A0[:, 2] = np.mean(Y, axis=0) - np.mean(X, axis=0)

    elif tform == TFORM.EUCLIDEAN:

        def ortho_err(Ap):
            R = AffineParams.R(Ap)
            return np.sum(R @ R.T - np.eye(2))

        # constraint: transpose = inverse
        constraints = opt.NonlinearConstraint(ortho_err, lb=0, ub=0)

        # initial guess from shape's relative angle, scale, and position:

        v1 = X[-1] - X[0]
        v2 = Y[-1] - Y[0]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        cos = np.dot(v1, v2) / (n1 * n2)
        sin = np.sqrt(1 - cos ** 2)

        A0 = np.zeros((2, 3))
        A0[:2, :2] = n2 / n1 * np.array([[cos, sin], [-sin, cos]])
        A0[:, 2] = np.mean(Y, axis=0) - np.mean(X, axis=0)

    else:
        raise ValueError("Unknown transformation type")

    # optimize using constraints and initial guess:

    Ap0 = AffineParams.from_mtrx(A0)
    objective = PointAlignmentObjective(X, Y, avg_tol=tol)

    opt_result = opt.minimize(objective, Ap0, method='trust-constr', callback=objective.converged,
                              constraints=constraints, options={'maxiter': maxiter},
                              hess=lambda _: np.zeros((len(Ap0), len(Ap0))))

    A = AffineParams.to_mtrx(opt_result.x)
    return A, objective.Xt, opt_result


class AffineParams:
    """
    Static class for converting between
    """

    @staticmethod
    def apply(Ap, X):
        return Ap[:4].reshape(2, 2) @ X + Ap[4:].reshape(-1, 1)

    @staticmethod
    def from_mtrx(A):
        return np.r_[A[:2, :2].flatten(), A[:2, 2]]

    @staticmethod
    def to_mtrx(Ap):
        return np.c_[Ap[:4].reshape(2, 2), Ap[4:].reshape(2, 1)]

    @staticmethod
    def R(Ap):
        return Ap[:4].reshape(2, 2)


class PointAlignmentObjective:

    def __init__(self, X, Y, avg_tol=1e-3):
        self.X = X.copy().T
        self.Y = Y.copy().T
        self.value = None
        self.Xt = None
        self.tol = avg_tol * len(X)

    def __call__(self, Ap):
        self.Xt = AffineParams.apply(Ap, self.X)
        self.value = np.sum((self.Xt - self.Y) ** 2)
        return self.value

    def converged(self, _, result):
        return self.value < self.tol and result.constr_violation < 1e-8
