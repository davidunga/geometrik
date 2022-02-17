import numpy as np
from utils import winding_angle, derivative
import geometrik as gk
from test_utils import *
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import cumulative_trapezoid
from enum import Enum
matplotlib.use("MacOSX")


class CURVE_REP(Enum):
    """ Curve representations """
    ANGLE_PROFILE = 'ANGLE_PROFILE'     # theta(arc-length)
    RADIUS_PROFILE = 'RADIUS_PROFILE'   # radius(theta)
    CARTESIAN = 'CARTESIAN'             # [x(i),y(i)]

    def __str__(self):
        return self.value


CURVE_REPRESENTATIONS = [cr for cr in CURVE_REP]

VARNAMES = {
    CURVE_REP.ANGLE_PROFILE: ('s', 't'),    # theta[arc-length]
    CURVE_REP.RADIUS_PROFILE: ('t', 'r'),   # radius[theta]
    CURVE_REP.CARTESIAN: ('x', 'y')         # (x[i], y[i])
}


class InvalidCurveRep(Exception):
    pass


def check_rep(z, rep: CURVE_REP):
    """
    Check curve representation
    """
    _eps = 1e-9
    if rep == CURVE_REP.ANGLE_PROFILE:
        s, t = z[:, 0], z[:, 1]
        if np.any(s < 0):
            raise InvalidCurveRep(f"{rep}: Arc-length must be non-negative")
        if np.min(s) > _eps:
            raise InvalidCurveRep(f"{rep}: Arc-length must start at zero")
        if np.any(np.diff(np.sign(np.diff(s)))):
            raise InvalidCurveRep(f"{rep}: Arc-length must be monotonic")
    elif rep == CURVE_REP.RADIUS_PROFILE:
        t, r = z[:, 0], z[:, 1]
        if np.any(np.abs(r) < _eps):
            raise InvalidCurveRep(f"{rep}: Radius must be non-zero")
        if np.any(np.diff(np.sign(np.diff(t)))):
            raise InvalidCurveRep(f"{rep}: Angle must be monotonic")


from abc import ABC, abstractmethod, abstractproperty


class Curve(ABC):

    def __init__(self):
        if isinstance(c, Curve):
            c = c.get(rep)
        check_rep(c, rep)
        self.c = c
        self.rep = rep
        self.varnames = VARNAMES[rep]

    def xy(self):
        return self.get(CURVE_REP.CARTESIAN)

    def get(self, rep: CURVE_REP = None):
        if rep is None:
            return self.c
        else:
            return convert(self.c, rep_from=self.rep, rep_to=rep)


class RadiusProfile:

    def __init__(self, theta, radius):
        self.t = theta
        self.r = radius

    def xy(self):
        return convert((self.t, self.r), rep_from=CURVE_REP.CARTESIAN)






def convert(c, rep_from: CURVE_REP, rep_to: CURVE_REP):
    return conversion_fnc(rep_from, rep_to)(c)


def conversion_fnc(rep_from: CURVE_REP, rep_to: CURVE_REP):
    if rep_from == rep_to:
        return lambda x: x
    return {
        (CURVE_REP.ANGLE_PROFILE, CURVE_REP.CARTESIAN): st2xy,
        (CURVE_REP.ANGLE_PROFILE, CURVE_REP.RADIUS_PROFILE): st2tr,
        (CURVE_REP.RADIUS_PROFILE, CURVE_REP.CARTESIAN): tr2xy,
        (CURVE_REP.RADIUS_PROFILE, CURVE_REP.ANGLE_PROFILE): tr2st,
        (CURVE_REP.CARTESIAN, CURVE_REP.RADIUS_PROFILE): xy2tr,
        (CURVE_REP.CARTESIAN, CURVE_REP.ANGLE_PROFILE): xy2st
    }[(rep_from, rep_to)]


def tr2xy(tr):
    """
    r(t) -> (x,y)
    Convert radius-profile to Cartesian
    """
    check_rep(tr, CURVE_REP.RADIUS_PROFILE)
    return st2xy(tr2st(tr))


def xy2tr(X):
    """
    (x,y) -> r(t)
    Convert Cartesian to radius-profile
    """
    t = winding_angle(X)
    r = np.abs(gk.euclidean_curvature(X) ** -1) * np.sign(t)
    tr = np.stack([t, r], axis=1)
    check_rep(tr, CURVE_REP.RADIUS_PROFILE)
    return tr


def st2tr(st):
    """
    t(s) -> r(t)
    Convert angle-profile to radius-profile
    """
    check_rep(st, CURVE_REP.ANGLE_PROFILE)
    s, t = st[:, 0], st[:, 1]
    r = derivative(t, s)[0]
    tr = np.stack([t, r], axis=1)
    check_rep(tr, CURVE_REP.RADIUS_PROFILE)
    return tr


def tr2st(tr):
    """
     r(t) -> t(s)
     Convert radius-profile to angle-profile
     """
    check_rep(tr, CURVE_REP.RADIUS_PROFILE)
    t, r = tr[:, 0], tr[:, 1]
    s = np.abs(cumulative_trapezoid(r ** -1, t, initial=0))
    t *= np.sign(r) * np.sign(t)
    st = np.stack([s, t], axis=1)
    check_rep(st, CURVE_REP.ANGLE_PROFILE)
    return st


def xy2st(X):
    """
    (x,y) -> t(s)
    Convert Cartesian to angle-profile
    """
    s = gk.euclidean_arclen(X)
    t = winding_angle(X)
    st = np.stack([s, t], axis=1)
    check_rep(st, CURVE_REP.ANGLE_PROFILE)
    return st


def st2xy(st):
    """
    t(s) -> (x,y)
    Convert angle-profile to Cartesian
    """
    check_rep(st, CURVE_REP.ANGLE_PROFILE)
    s, t = st[:, 0], st[:, 1]
    ds = np.diff(s)
    ds = np.r_[ds, 2 * ds[-1] - ds[-2]]
    X = np.cumsum(np.stack([ds * np.cos(t), ds * np.sin(t)], axis=1), axis=0)
    return X


def test_consistency():
    from itertools import product
    import warnings
    warnings.filterwarnings("error")

    errs = []
    show = False
    t = np.linspace(np.pi/16, .5 * 15 * np.pi/16, 500)
    ix = 0
    for sgn1, sgn2, d0 in product((-1, 1), (-1, 1), (False, True)):

        for repr_from in CURVE_REPRESENTATIONS:
            for repr_to in CURVE_REPRESENTATIONS:
                if repr_to == repr_from:
                    continue

                c = np.stack([sgn1 * t, sgn2 * t ** 2], axis=1)
                if d0:
                    c[:, 0] -= c[0, 0]

                if repr_from == CURVE_REP.CARTESIAN:
                    c -= c[0]

                try:
                    check_rep(c, repr_from)
                except InvalidCurveRep as e:
                    print(e)
                    continue

                print(f"{repr_from} -> {repr_to}", end="")
                c_converted = convert(c, repr_from, repr_to)

                print(f" -> {repr_from}", end="")
                c_reconst = convert(c_converted, repr_to, repr_from)

                err = np.abs(c - c_reconst)
                errs.append(np.max(err))

                print(" Error={:2.2f}".format(err.mean()))

                if not show:
                    continue

                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
                fig.set_size_inches(10, 5)

                ax1.plot(c[:, 0], c[:, 1], 'ko')
                ax1.plot(c[0, 0], c[0, 1], 'bs')
                ax1.plot(c_reconst[:, 0], c_reconst[:, 1], 'r')
                ax1.plot(c_reconst[0, 0], c_reconst[0, 1], 'rs')
                ax1.set_xlabel(VARNAMES[repr_from][0])
                ax1.set_ylabel(VARNAMES[repr_from][1])
                ax1.set_title(str(repr_from))

                ax2.plot(c_converted[:, 0], c_converted[:, 1], ':r')
                ax2.plot(c_converted[0, 0], c_converted[0, 1], 'sr')
                ax2.set_xlabel(VARNAMES[repr_to][0])
                ax2.set_ylabel(VARNAMES[repr_to][1])
                ax2.set_title(str(repr_to))

                plt.show()

    print(np.max(errs))



if __name__ == "__main__":
    test_consistency()