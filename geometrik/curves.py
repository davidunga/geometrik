"""
Curve Representations.
Allows parameterizing curves as Cartesian, Radius-Profile, or Angle-Profile, and
    converting between the representations.
"""

from geometrik.utils import winding_angle, derivative
from geometrik.measures import euclidean_arclen, euclidean_curvature
from scipy.integrate import cumulative_trapezoid
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

class InvalidCurve(Exception):
    pass


class Curve(ABC):

    """
    Initialized either by keyword args, by another curve instance, or by
        sampled curve coordinates (Nx2 np array).
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
        elif isinstance(arg, np.ndarray):
            assert arg.ndim == 2
            assert arg.shape[1] == 2
            self._fromCartesian(Cartesian(xy=arg))
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
    Initialize by xy coordinates as a Nx2 ndarray, or by another curve.
        e.g.
        # by ndarray:
        t = np.linspace(0, 1, 500)
        xy = np.stack([t, t ** 2], axis=1)
        Cartesian(xy=xy)
        # by converting another curve:
        # ap = AngleProfile(..)
        Cartesian(ap)
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
        self.s = euclidean_arclen(cr.xy)
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
        self.r = euclidean_curvature(cr.xy) ** -1
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
