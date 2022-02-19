"""
Curve Representations.
Allows parameterizing curves as Cartesian, Radius-Profile, or Angle-Profile, and
    converting between the representations.
"""

import numpy as np
from utils import winding_angle, derivative
import geometrik as gk
from test_utils import *
from scipy.integrate import cumulative_trapezoid
from abc import ABC, abstractmethod
from copy import deepcopy


class InvalidCurve(Exception):
    pass


class Curve(ABC):

    """
    Initialized either by keyword args, or by another curve
    """

    def __init__(self, *args, **kwargs):
        assert (len(args) > 0) != (len(kwargs) > 0)
        if len(args) == 0:
            arg = deepcopy(kwargs)
        else:
            arg = deepcopy(args[0])
        self._construct(arg)

    def _construct(self, arg):
        if isinstance(arg, Cartesian):
            self._fromCartesian(arg)
        elif isinstance(arg, AngleProfile):
            self._fromAngleProfile(arg)
        elif isinstance(arg, RadiusProfile):
            self._fromRadiusProfile(arg)
        elif isinstance(arg, dict):
            self._fromDict(arg)
        else:
            raise TypeError(f"Cannot instantiate from type {type(arg)}")
        self._check()

    def xy(self):
        """ Cartesian coordinates of curve """
        return Cartesian(self).xy

    @abstractmethod
    def as_np(self):
        pass

    def _fromDict(self, d):
        for k in d:
            self.__setattr__(k, d[k])

    @abstractmethod
    def _fromCartesian(self, cr):
        pass

    @abstractmethod
    def _fromAngleProfile(self, ap):
        pass

    @abstractmethod
    def _fromRadiusProfile(self, rp):
        pass

    def _check(self):
        pass


class Cartesian(Curve):

    """
    Initialize by supplying xy coordinates as a Nx2 ndarray, or by converting
        another curve, e.g.
        Cartesian(xy=np.rand(500,2))
        Cartesian(curve_instance)
    """

    def __init__(self, *args, **kwargs):
        self.xy = None
        super().__init__(*args, **kwargs)

    def as_np(self):
        return self.xy

    def _fromCartesian(self, cr):
        self.xy = cr.xy

    def _fromAngleProfile(self, ap):
        s, t = ap.s, ap.t
        ds = np.diff(s)
        ds = np.r_[ds, 2 * ds[-1] - ds[-2]]
        self.xy = np.cumsum(np.stack([ds * np.cos(t), ds * np.sin(t)], axis=1), axis=0)

    def _fromRadiusProfile(self, rp):
        dt = np.diff(rp.t)
        dt = np.r_[dt, dt[-1]]
        rho = 2 * np.sin(.5 * dt) / rp.r
        dxy = np.stack([rho * np.cos(rp.t), rho * np.sin(rp.t)], axis=1)
        self.xy = np.cumsum(dxy, axis=0)

    def _check(self):
        assert self.xy is not None


class AngleProfile(Curve):
    """
    Curve representation as theta(arc-length).
    Arc-length must be monotonically increasing, non-negative.

    Initialize by supplying theta and arc-length values, (keywords: t & s), or
    by converting another curve, e.g.
        # By parameters:
        theta = np.linspace(np.pi, 500)
        arc_length = np.linspace(0, 1, 500)
        AngleProfile(t=theta, s=arc_length)
        # By conversion from Cartesian:
        AngleProfile(Cartesian(xy=np.rand(500,2))
    """

    def __init__(self, *args, **kwargs):
        self.s, self.t = None, None
        super().__init__(*args, **kwargs)

    def as_np(self):
        return np.stack([self.s, self.t], axis=1)

    def _fromCartesian(self, cr):
        self.s = gk.euclidean_arclen(cr.xy)
        self.t = winding_angle(cr.xy)

    def _fromAngleProfile(self, ap):
        self.s, self.t = ap.s, ap.t

    def _fromRadiusProfile(self, rp):
        dt = np.diff(rp.t)
        dt = np.r_[dt, dt[-1]]
        self.s = np.abs(cumulative_trapezoid(rp.r ** -1, rp.t, initial=0))
        self.t = rp.t
        if rp.t[-1] < rp.t[0]:
            self.t += np.pi

    def _check(self):
        assert self.s is not None
        assert self.t is not None
        if np.any(self.s < 0):
            raise InvalidCurve("AngleProfile: Arc-length must be non-negative")
        if np.min(self.s) > 1e-9:
            raise InvalidCurve("AngleProfile: Arc-length must start at zero")
        if np.any(np.diff(np.sign(np.diff(self.s)))):
            raise InvalidCurve("AngleProfile: Arc-length must be monotonic")


class RadiusProfile(Curve):
    """
    Curve representation as radius(theta)
    Radius must be positive
    Theta must be monotonically increasing / decreasing (convex)

    Initialize by supplying theta and radius values, (keywords: t & r), or
    by converting another curve, e.g.
        theta = np.linspace(np.pi, 500)
        rads = np.linspace(0, 1, 500)
        RadiusProfile(t=theta, r=rads)
    """

    def __init__(self, *args, **kwargs):
        self.t, self.r = None, None
        super().__init__(*args, **kwargs)

    def as_np(self):
        return np.stack([self.t, self.r], axis=1)

    def _fromCartesian(self, cr):
        self.t = winding_angle(cr.xy)
        self.r = gk.euclidean_curvature(cr.xy) ** -1
        if self.r[0] < 0:
            self.r *= -1
            self.t += np.sign(self.t[-1] - self.t[0]) * np.pi

    def _fromAngleProfile(self, ap):
        self.r = derivative(ap.t, ap.s)[0]
        self.t = ap.t
        if self.r[0] < 0:
            self.r *= -1
            self.t += np.sign(ap.t[-1] - ap.t[0]) * np.pi

    def _fromRadiusProfile(self, rp):
        self.t, self.r = rp.t, rp.r

    def _check(self):
        assert self.r is not None
        assert self.t is not None
        if np.any(self.r < 1e-9):
            raise InvalidCurve("RadiusProfile: Radius must be positive")
        if len(np.unique(np.sign(np.diff(self.t)))) > 1:
            raise InvalidCurve("RadiusProfile: Angle must be monotonic")


def test_consistency():
    """
    Test that converting a curve to a different curve, and then back again,
    yields the original curve.
    """

    _err_thresh = 0.05

    xy, _ = get_shape_points('ellipse')
    n = 500
    r = np.linspace(.01, 2, n) ** 2
    t = np.linspace(.01, np.pi - .01, n)
    s = np.linspace(0, 5, n)

    curves = [
        RadiusProfile(t=t, r=r),
        RadiusProfile(t=t[::-1], r=r),
        RadiusProfile(t=t, r=r[::-1]),
        RadiusProfile(t=t[::-1], r=r[::-1]),
        AngleProfile(s=s, t=t),
        AngleProfile(s=s, t=t[::-1]),
        Cartesian(xy=xy - xy[0]),
        Cartesian(xy=np.fliplr(xy - xy[0])),
        Cartesian(xy=np.flipud(xy - xy[-1])),
    ]

    error_reports = []
    for curve in curves:
        for cnvrt_type in [RadiusProfile, AngleProfile, Cartesian]:
            src_type = type(curve)

            curve_converted = cnvrt_type(curve)
            curve_reconstructed = src_type(curve_converted)

            if src_type == Cartesian:
                err = np.max(np.abs(curve.as_np() - curve_reconstructed.as_np()))
            else:
                dt = curve.t - curve_reconstructed.t
                if dt[0] < 0:
                    dt *= -1
                dt -= np.round(dt[0] / (2 * np.pi)) * (2 * np.pi)
                dt_err = np.max(np.abs(dt))
                if src_type == RadiusProfile:
                    err = np.max([dt_err, np.max(np.abs(curve.r - curve_reconstructed.r))])
                else:
                    err = np.max([dt_err, np.max(np.abs(curve.s - curve_reconstructed.s))])

            report = f"{src_type} -> {cnvrt_type} -> {src_type}" + " Error={:2.2f}".format(err)
            print(report)

            if err > _err_thresh:
                error_reports.append(report)

    print("Conversion consistency test done.")
    if len(error_reports) == 0:
        print("No issues we found.")
    else:
        print("Major errors:")
        for error_report in error_reports:
            print(error_report)


if __name__ == "__main__":
    test_consistency()
