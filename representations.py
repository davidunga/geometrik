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

    def __init__(self, arg, param_names):
        self.param_names = param_names
        self._orientation = None
        self._construct(deepcopy(arg))

    def _construct(self, arg):
        if isinstance(arg, Cartesian):
            self._fromCartesian(arg)
        elif isinstance(arg, AngleProfile):
            self._fromAngleProfile(arg)
        elif isinstance(arg, RadiusProfile):
            self._fromRadiusProfile(arg)
        elif isinstance(arg, (list, tuple)):
            self._fromParams(arg)
        elif isinstance(arg, np.ndarray):
            self._fromNumpy(arg)
        elif isinstance(arg, dict):
            self._fromDict(arg)
        else:
            raise TypeError(f"Cannot instantiate from type {type(arg)}")
        self._check()

    def xy(self):
        return Cartesian(self).xy

    @abstractmethod
    def as_np(self):
        pass

    def _fromDict(self, d):
        for k in d:
            self.__setattr__(k, d[k])

    @abstractmethod
    def _fromParams(self, p):
        pass

    @abstractmethod
    def _fromNumpy(self, p):
        pass

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

    def __init__(self, arg):
        self.xy = None
        super().__init__(arg, ['x', 'y'])

    def as_np(self):
        return self.xy

    def _fromParams(self, p):
        self.xy = np.stack(p, axis=1)

    def _fromNumpy(self, p):
        self.xy = np.copy(p)

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
    Curve representation as theta(arc-length)
        arc-length must be monotonically increasing, non-negative
    """

    def __init__(self, arg):
        self.s, self.t = None, None
        super().__init__(arg, ['arclen', 'theta'])

    def as_np(self):
        return np.stack([self.s, self.t], axis=1)

    def _fromParams(self, p):
        self.s, self.t = p

    def _fromNumpy(self, p):
        self.s, self.t = p[:, 0], p[:, 1]

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
        radius must be positive
        theta must be monotonically increasing / decreasing
    """

    def __init__(self, arg):
        self.t, self.r = None, None
        super().__init__(arg, ['theta', 'radius'])

    def as_np(self):
        return np.stack([self.t, self.r], axis=1)

    def _fromParams(self, p):
        self.t, self.r = p

    def _fromNumpy(self, p):
        self.t, self.r = p[:, 0], p[:, 1]

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
    Test that converting a curve to a different curve, and then back again, results in the
    original curve.
    """

    _err_thresh = 0.05

    xy, _ = get_shape_points('ellipse')
    n = 500
    r = np.linspace(.01, 2, n) ** 2
    t = np.linspace(.01, np.pi - .01, n)
    s = np.linspace(0, 5, n)

    curves = [
        RadiusProfile(dict(t=t, r=r)),
        RadiusProfile(dict(t=t[::-1], r=r)),
        RadiusProfile(dict(t=t, r=r[::-1])),
        RadiusProfile(dict(t=t[::-1], r=r[::-1])),
        AngleProfile(dict(s=s, t=t)),
        AngleProfile(dict(s=s, t=t[::-1])),
        Cartesian(dict(xy=xy - xy[0])),
        Cartesian(dict(xy=np.fliplr(xy - xy[0]))),
        Cartesian(dict(xy=np.flipud(xy - xy[-1]))),
    ]

    error_reports = []
    for curve in curves:
        for cnvrt_type in [RadiusProfile, AngleProfile, Cartesian]:
            src_type = type(curve)
            if src_type == cnvrt_type:
                continue

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

            is_error = np.max(err) > _err_thresh
            if is_error:
                error_reports.append(report)

            show = True
            if show and is_error:
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
                fig.set_size_inches(10, 5)

                c = curve.as_np()
                cr = curve_reconstructed.as_np()
                cc = curve_converted.as_np()

                ax1.plot(c[:, 0], c[:, 1], 'ko')
                ax1.plot(c[0, 0], c[0, 1], 'bs')
                ax1.plot(cr[:, 0], cr[:, 1], 'r')
                ax1.plot(cr[0, 0], cr[0, 1], 'rs')
                ax1.set_xlabel(curve.param_names[0])
                ax1.set_ylabel(curve.param_names[1])
                ax1.set_title(src_type)

                ax2.plot(cc[:, 0], cc[:, 1], ':r')
                ax2.plot(cc[0, 0], cc[0, 1], 'sr')
                ax2.set_xlabel(curve_converted.param_names[0])
                ax2.set_ylabel(curve_converted.param_names[1])
                ax2.set_title(cnvrt_type)

                plt.show()

    print("Conversion consistency test done.")
    if len(error_reports) == 0:
        print("No major errors we found.")
    else:
        print("Major errors:")
        for error_report in error_reports:
            print(error_report)


if __name__ == "__main__":
    test_consistency()
