import numpy as np
from utils import winding_angle, derivative
import geometrik as gk
from test_utils import *
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import cumulative_trapezoid
from abc import ABC, abstractmethod
from copy import deepcopy

matplotlib.use("MacOSX")


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


def test_1():

    a = np.linspace(np.pi / 16, .5 * 15 * np.pi / 16, 500)

    s = a - a[0]
    curves = {
        'AP_reg': RadiusProfile(AngleProfile((s, a))),
        'AP_flip_t': RadiusProfile(AngleProfile((s, a[::-1])))
    }
    curves = {
        'RP_reg': AngleProfile(RadiusProfile((a, a))),
        'RP_flip_t': AngleProfile(RadiusProfile((a[::-1], a))),
        'RP_minus_t': AngleProfile(RadiusProfile((-a, a))),
        'RP_flip_t_minus_r': AngleProfile(RadiusProfile((-a[::-1], a))),
    }
    keys = list(curves.keys())
    for i in range(len(keys) - 1):
        for j in range(i + 1, len(keys)):
            err = np.abs(curves[keys[i]].as_np() - curves[keys[j]].as_np()).mean()
            if err < 0.1:
                print(f"Cannot distinguish: {keys[i]} - {keys[j]}")
    print(".")


def test_consistency():
    from itertools import product
    import warnings
    warnings.filterwarnings("error")

    curve_types = [Cartesian, AngleProfile, RadiusProfile]

    z = np.linspace(np.pi / 16, .5 * 15 * np.pi / 16, 500)

    curves = [
        RadiusProfile(dict(t=z, r=z**2)),
        RadiusProfile(dict(t=z[::-1], r=z ** 2)),
        RadiusProfile(dict(t=z[::-1], r=z[::-1] ** 2)),
        RadiusProfile(dict(t=z, r=z[::-1] ** 2)),
    ]
    
    errs = []
    show = True
    err_report = []
    err_params = []
    t = np.linspace(np.pi/16, .5 * 15 * np.pi/16, 500)
    for sgn1, sgn2, d0 in product((1, -1), (1, -1), (False, True)):

        for type_from in curve_types:
            for type_to in curve_types:

                if type_from == type_to:
                    continue

                c = np.stack([t if sgn1 > 0 else t[::-1], sgn2 * np.sin(t)], axis=1)
                if d0:
                    c[:, 0] -= c[0, 0]

                if type_from == Cartesian:
                    c -= c[0]

                try:
                    curve = type_from(c)
                except InvalidCurve as e:
                    print(e)
                    continue

                assert np.max(np.abs(c - curve.as_np())) < 1e-9

                print(f"{type_from_str} -> {type_to_str}", end="")
                curve_converted = type_to(curve)

                print(f" -> {type_from_str}", end="")
                curve_reconst = type_from(curve_converted)

                try:
                    rd = (abs(curve.t[0] - curve_reconst.t[0]) / np.pi)
                    #print("delta t: ", abs(round(rd) - rd))
                except:
                    pass
                err = np.abs(c - curve_reconst.as_np())
                errs.append(np.max(err))

                print(" Error={:2.2f}".format(err.max()))

                #show = False
                if np.max(err) > 0.01:
                    show = True
                    err_params.append((sgn1, sgn2, d0))
                    err_report.append(f"{type_from_str} -> {type_to_str} -> {type_from_str}" + " Error={:2.2f}".format(np.max(err)))

                if not show:
                    continue

                c_reconst = curve_reconst.as_np()
                c_converted = curve_converted.as_np()

                fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
                fig.set_size_inches(10, 5)

                ax1.plot(c[:, 0], c[:, 1], 'ko')
                ax1.plot(c[0, 0], c[0, 1], 'bs')
                ax1.plot(c_reconst[:, 0], c_reconst[:, 1], 'r')
                ax1.plot(c_reconst[0, 0], c_reconst[0, 1], 'rs')
                ax1.set_xlabel(curve.param_names[0])
                ax1.set_ylabel(curve.param_names[1])
                ax1.set_title(type_from_str)

                ax2.plot(c_converted[:, 0], c_converted[:, 1], ':r')
                ax2.plot(c_converted[0, 0], c_converted[0, 1], 'sr')
                ax2.set_xlabel(curve_converted.param_names[0])
                ax2.set_ylabel(curve_converted.param_names[1])
                ax2.set_title(type_to_str)

                plt.show()

    print("---")
    for i in range(len(err_report)):
        print(err_params[i], err_report[i])


def test_2():
    t = np.linspace(np.pi/16, .5 * 15 * np.pi/16, 500)
    s = t ** 2
    s -= s[0]
    curves = {
        'reg': AngleProfile(dict(s=s, t=-t)),
        't_flip': AngleProfile(dict(s=s, t=-t[::-1]))
    }

    for k in curves:
        curve = curves[k]
        c = curve.as_np()
        crt_curve = Cartesian(curve)
        cc = crt_curve.as_np()
        c_recon = AngleProfile(crt_curve).as_np()

        print("theta diff = {:2.2f}".format(c_recon[0,0] - c[0,0]))

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
        fig.set_size_inches(10, 5)

        ax1.plot(c[:, 0], c[:, 1], 'ko')
        ax1.plot(c[0, 0], c[0, 1], 'bs')
        ax1.plot(c_recon[:, 0], c_recon[:, 1], 'r-')
        ax1.plot(c_recon[0, 0], c_recon[0, 1], 'rs')
        ax1.set_xlabel(curve.param_names[0])
        ax1.set_ylabel(curve.param_names[1])
        ax2.set_title(curve)

        ax2.plot(cc[:, 0], cc[:, 1], ':r')
        ax2.plot(cc[0, 0], cc[0, 1], 'sr')
        ax2.set_xlabel(crt_curve.param_names[0])
        ax2.set_ylabel(crt_curve.param_names[1])
        ax2.set_title(crt_curve)
        plt.suptitle(k)

        plt.show()



if __name__ == "__main__":
    #test_2()
    test_1()
    test_consistency()
