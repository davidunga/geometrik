import numpy as np
from utils import winding_angle, derivative
import geometrik as gk
from test_utils import *
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from enum import Enum


class CURVE_REPR(Enum):
    """ Curve representations """
    ANGLE_PROFILE = 'ANGLE_PROFILE'     # theta(arc_length)
    RADIUS_PROFILE = 'RADIUS_PROFILE'   # radius(theta)
    CARTESIAN = 'CARTESIAN'             # [x(i),y(i)]

    def __str__(self):
        return self.value

CURVE_REPRESENTATIONS = [cr for cr in CURVE_REPR]

VARNAMES = {
    CURVE_REPR.ANGLE_PROFILE: ('arclen', 'theta'),
    CURVE_REPR.RADIUS_PROFILE: ('theta', 'radius'),
    CURVE_REPR.CARTESIAN: ('x', 'y')
}


class Curve:

    def __init__(self, c, rep: CURVE_REPR):
        if isinstance(c, Curve):
            c = c.get(rep)
        self.c = np.stack(c, axis=1)
        self.rep = rep
        self.varnames = VARNAMES[rep]

    def get(self, rep: CURVE_REPR = None):
        if rep is None:
            return self.c
        else:
            return convert(self.c, rep_from=self.rep, rep_to=rep)


def convert(c, rep_from: CURVE_REPR, rep_to: CURVE_REPR):
    if rep_from == rep_to:
        return c
    return {
        (CURVE_REPR.ANGLE_PROFILE, CURVE_REPR.CARTESIAN): ts_to_xy,
        (CURVE_REPR.ANGLE_PROFILE, CURVE_REPR.RADIUS_PROFILE): ts_to_rt,
        (CURVE_REPR.RADIUS_PROFILE, CURVE_REPR.CARTESIAN): rt_to_xy,
        (CURVE_REPR.RADIUS_PROFILE, CURVE_REPR.ANGLE_PROFILE): rt_to_ts,
        (CURVE_REPR.CARTESIAN, CURVE_REPR.RADIUS_PROFILE): xy_to_rt,
        (CURVE_REPR.CARTESIAN, CURVE_REPR.ANGLE_PROFILE): xy_to_ts
    }[(rep_from, rep_to)](c)


def rt_to_xy(rt):
    """ Convert radius-theta to xy """
    return ts_to_xy(rt_to_ts(rt))


def xy_to_rt(X):
    """ Convert xy to radius-theta """
    t = winding_angle(X)
    r = gk.euclidean_curvature(X) ** -1
    return np.stack([r, t], axis=1)


def ts_to_rt(ts):
    t, s = ts[:, 0], ts[:, 1]
    r = np.gradient(s, t)
    return np.stack([r, t], axis=1)


def rt_to_ts(rt):
    r, t = rt[:, 0], rt[:, 1]
    s = cumulative_trapezoid(r, t, initial=0)
    return np.stack([t, s], axis=1)


def xy_to_ts(X):
    """ Convert xy to theta-arclength (angle profile) """
    s = gk.euclidean_arclen(X)
    t = winding_angle(X)
    return np.stack([t, s], axis=1)


def ts_to_xy(ts):
    """ Convert theta-arclength (angle profile) to xy """
    t, s = ts[:, 0], ts[:, 1]
    ds = np.diff(s)
    ds = np.r_[ds, 2 * ds[-1] - ds[-2]]
    X = np.cumsum(np.stack([ds * np.cos(t), ds * np.sin(t)], axis=1), axis=0)
    return X


def test_consistency():
    import warnings
    warnings.filterwarnings("error")

    t = np.linspace(np.pi/16, .5 * 15 * np.pi/16, 500)
    for repr_from in CURVE_REPRESENTATIONS:
        for repr_to in CURVE_REPRESENTATIONS:
            if repr_to == repr_from:
                continue

            c = np.stack([t, -t ** 2], axis=1)

            print(f"{repr_from} -> {repr_to}", end="")
            c_converted = convert(c, repr_from, repr_to)

            print(f" -> {repr_from}", end="")
            c_reconst = convert(c_converted, repr_to, repr_from)

            c -= np.mean(c, axis=0)
            c_reconst -= np.mean(c_reconst, axis=0)

            print(" Error={:2.2f}".format(np.abs(c - c_reconst).mean()))

            plt.plot(c[:, 0], c[:, 1], 'ko')
            plt.plot(c_converted[:, 0], c_converted[:, 1], 'r')
            plt.plot(c_reconst[:, 0], c_reconst[:, 1], 'b')
            plt.waitforbuttonpress()
            plt.cla()





def test_st():

    X, _ = get_shape_points('ellipse')

    s, t = xy_to_st(X)
    XX = st_to_xy(s, t)
    ss, tt = xy_to_st(XX)

    X -= np.mean(X, axis=0)
    XX -= np.mean(XX, axis=0)

    s_err = np.max(np.abs(s - ss))
    t_err = np.max(np.abs(t - tt))
    X_err = np.mean(np.abs(X - XX))
    print("Errors: s={:2.2f} t={:2.2f} X={:2.2f}".format(s_err, t_err, X_err))

    plt.plot(t, s, 'ro')
    plt.plot(tt, ss, 'b')
    plt.show()

    plt.plot(X[:, 0], X[:, 1], 'r', X[0, 0], X[0, 1], 'ro')
    plt.plot(XX[:, 0], XX[:, 1], 'b', XX[0, 0], XX[0, 1], 'bs')
    plt.show()

if __name__ == "__main__":

    test_consistency()