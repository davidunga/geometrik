"""

Implementing log-radius profile representation of convex curves described in:
1. Huh, D. (2015). The vector space of convex curves: How to mix shapes.
    arXiv preprint arXiv:1506.07515.
2. Huh, D., & Sejnowski, T. J. (2015). Spectrum of power laws for curved hand
movements. Proceedings of the National Academy of Sciences, 112(29), E3950-E3958.

This representation facilitates an algebra of shapes, in the sense that for two
shapes h1 & h2, represented in this manner, their weighted sum: h3 = h1 + s*h2
(s ~real) is a visually intuitive mixture of h1 & h2.

This representation is also tightly related with how the brain generates curved
movements. Specifically, it provides insight regarding the power-law relation
between speed and curvature in human movement. (See [2]).

"""

import numpy as np
from dataclasses import dataclass
from curves import RadiusProfile, Curve
from utils import fourier, angdiff
from fractions import Fraction


@dataclass
class HuhParams:
    # parameters of pure shape.
    m: int              # rotational symmetry (number of max-curvature points)
    n: int              # period size
    eps: float          # amplitude
    t0: float = None    # phase (None = use shape's "native" phase)

    def __post_init__(self):
        assert self.n != 0
        assert np.gcd(self.m, self.n) == 1, "n & m must be co-prime"
        self.nu = self.m / self.n
        if self.t0 is None:
            if self.m > 0:
                # phase = curvature maxima + one-part of rotation
                self.t0 = -.5 * np.pi / self.nu - (1 / self.m) * np.pi
            else:
                self.t0 = 0

    def isclose(self, other):
        """
        Are params equal-upto-tolerance
        """
        assert isinstance(other, HuhParams)
        _eps_rtol = .01             # error tolerance for eps param: 1%
        _t0_atol = np.deg2rad(.1)   # error tolerance for t0 param: 0.1 degree
        if self.n != other.n or self.m != other.m:
            return False
        if abs(self.eps - other.eps) > _eps_rtol * max(self.eps, other.eps):
            return False
        if angdiff(self.t0, other.t0) > _t0_atol:
            return False
        return True


class HuhCurve:

    def __init__(self, params):
        """
        Initialize Huh Curve.
        :param params: Either a HuhParams object, or a list of such objects.
        Examples:
            HuhCurve(HuhParams(m=3, n=2, eps=0.8))
            HuhCurve([HuhParams(m=6, n=1, eps=1.2), HuhParams(m=2, n=5, eps=0.8)])
        """
        if isinstance(params, HuhParams):
            params = [params]
        assert (isinstance(params, list) and
                all([isinstance(p, HuhParams) for p in params])),\
            "Expected a HuhParams object, or a list of HuhParams objects"
        self.params = [params[i] for i in np.argsort([p.nu for p in params])]

    def is_pure(self):
        return len(self.params) == 1

    def period(self):
        """ Get shape's period """
        _nonperiodic_n = 4  # period to use for open shapes
        return 2 * np.pi * np.prod([c.n if c.m != 1 else _nonperiodic_n for c in self.params])

    def full_period_thetas(self, res):
        """
        Make linearly spaced theta values to cover the full period.
        :param res: resolution. either number of samples (if int and > 1),
            or the sampling step in radians (if float and < 1)
        """
        assert (isinstance(res, int) and res > 1) or (isinstance(res, float) and 0 < res < 1),\
            "resolution must be either integer > 1, or positive float < 1"
        n = res if res > 1 else int(round(self.period() / res))
        return np.linspace(0, self.period(), n)

    def full_period_curve(self, res):
        """
        Get a radius-profile curve, covering the full shape's period
        :param res: Sampling resolution. See full_period_thetas()
        :return: RadiusProfile instance
        """
        t = self.full_period_thetas(res)
        return RadiusProfile(t=t, r=np.exp(self.log_r(t)))

    def log_r(self, t: np.ndarray):
        """
        log radius of curvature for given thetas
        :param t: 1d array of thetas (winding angles)
        :return: lgr[i] is the log-radius at the i-th point
        """
        lgr = 0
        for c in self.params:
            lgr += c.eps * np.sin(c.nu * (t - c.t0))
        return lgr

    def __add__(self, other):
        """ Add two shapes """
        return HuhCurve(self.params + other.params)

    def __rmul__(self, scale):
        """ Multiply shape by scalar """
        return HuhCurve([HuhParams(m=c.m, n=c.n, eps=scale * c.eps, t0=c.t0)
                         for c in self.params])

    def isclose(self, other):
        """
        Are curves equal-upto-tolerance
        :param other: other HuhCurve
        :return: boolean
        """
        assert isinstance(other, HuhCurve)
        if len(self.params) != len(other.params):
            return False
        return all([p.isclose(q) for p, q in zip(self.params, other.params)])

    @classmethod
    def fromCruve(cls, curve: (Curve, np.ndarray), explained=.9, max_components=5, max_n=50):
        """
        Estimate HuhShape from convex curve data
        :param curve: Either a curve instance or a sample points as Nx2 np array
        :param explained: part of power spectrum to explain
        :param max_components: max number of pure components
        * The number of components is: min(max_components, n_explained) where n_explained is the
            number of components required to explain [explained] of the power
        :param max_n: max value of 'n' (period parameter)
        """

        # represent curve as radius-profile:
        rp = RadiusProfile(curve)

        # fourier for radius-profile:
        fr, frq = fourier(np.log(np.abs(rp.r)), rp.t)

        FR = (np.abs(fr) / len(fr)) ** 2
        si = np.argsort(FR)[::-1]
        pwr_explained = np.cumsum(FR[si]) / np.sum(FR[si])

        if si[0] == 0 and pwr_explained[0] < explained:
            # if the first component is DC, but other components are also needed-
            # discard the DC. (i.e. we use the DC component only if its the only
            # component..)
            si = si[1:]
            pwr_explained = np.cumsum(FR[si]) / np.sum(FR[si])

        num_components = np.nonzero(pwr_explained >= explained)[0][0] + 1
        num_components = min(num_components, max_components)

        huh_params = []
        for ix in si[:num_components]:
            frc = Fraction(frq[ix]).limit_denominator(max_n)
            huh_params.append(HuhParams(m=frc.numerator,
                                        n=frc.denominator,
                                        eps=np.sqrt(FR[ix])))

        return cls(huh_params)

