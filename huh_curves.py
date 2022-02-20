"""

Implementing convex parametric curves described in:
1. Huh, D. (2015). The vector space of convex curves: How to mix shapes.
    arXiv preprint arXiv:1506.07515.
2. Huh, D., & Sejnowski, T. J. (2015). Spectrum of power laws for curved hand
movements. Proceedings of the National Academy of Sciences, 112(29), E3950-E3958.

"""

import numpy as np
from dataclasses import dataclass
from curve_repr import RadiusProfile, Curve
from utils import fourier, angdiff
from fractions import Fraction
import matplotlib.pyplot as plt

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
        :param res: resolution. either number of samples (>1), or the sampling step (<1)
        """
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
        if len(self.params) != len(other.params):
            return False
        return all([p.isclose(q) for p, q in zip(self.params, other.params)])

    @classmethod
    def fromCruve(cls, curve: (Curve, np.ndarray),
                  max_n=50, max_frqs=5, explained_th=.9):

        rp = RadiusProfile(curve)
        fr, frq = fourier(np.log(np.abs(rp.r)), rp.t)
        FR = (np.abs(fr) / len(fr)) ** 2
        si = np.argsort(FR)[::-1]
        pwr_explained = np.cumsum(FR[si]) / np.sum(FR[si])

        if si[0] == 0 and pwr_explained[0] < explained_th:
            # if the first component is DC, but other components are also needed-
            # discard the DC. (i.e. we use the DC component only if its the only
            # component..)
            si = si[1:]
            pwr_explained = np.cumsum(FR[si]) / np.sum(FR[si])

        num_frqs = np.nonzero(pwr_explained >= explained_th)[0][0] + 1
        num_frqs = min(num_frqs, max_frqs)

        huh_params = []
        for ix in si[:num_frqs]:
            frc = Fraction(frq[ix]).limit_denominator(max_n)
            m = frc.numerator
            n = frc.denominator
            eps = np.sqrt(FR[ix])
            huh_params.append(HuhParams(m=m, n=n, eps=eps))

        return cls(huh_params)


def _plot(ax, h: HuhCurve, color):
    plt.sca(ax)
    X = h.full_period_curve(0.001).xy()
    plt.plot(X[:, 0], X[:, 1], color)
    plt.axis('square')


def demo():
    # pure shapes:
    h_a = HuhCurve(HuhParams(m=3, n=2, eps=1.0))
    h_b = HuhCurve(HuhParams(m=3, n=1, eps=1.2))
    h_c = HuhCurve(HuhParams(m=6, n=1, eps=1.6))

    # new shapes from pure shapes:
    h_ab = h_a + h_b
    h_ac = h_a + h_c

    # show:
    fig, axs = plt.subplots(ncols=3, nrows=2)
    fig.set_size_inches(10, 7)

    # --
    _plot(axs[0, 0], h_a, 'r')
    p = h_a.params[0]
    plt.title(f"[A] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[0, 1], h_b, 'r')
    p = h_b.params[0]
    plt.title(f"[B] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[0, 2], h_ab, 'r')
    plt.title(f"[A+B]")
    # --
    _plot(axs[1, 0], h_a, 'g')
    p = h_a.params[0]
    plt.title(f"[A] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[1, 1], h_c, 'g')
    p = h_c.params[0]
    plt.title(f"[C] m={p.m}, n={p.n} eps={p.eps}")

    _plot(axs[1, 2], h_ac, 'g')
    plt.title(f"[A+C]")
    # --
    plt.show()


def test():

    h = HuhCurve([HuhParams(m=5, n=1, eps=1.8),
                  HuhParams(m=5, n=3, eps=1.2)])
    curve = h.full_period_curve(0.01).xy()
    hr = HuhCurve.fromCruve(curve)

    print("Testing reconstruction of HuhCurve from xy samples.")
    print("Original curve:")
    for p in h.params:
        print(" ", p)
    print("Reconstructed curve:")
    for p in hr.params:
        print(" ", p)

    if hr.isclose(h):
        print("SUCCESSFUL reconstruction.")
    else:
        print("FAILED reconstruction.")

    _, axs = plt.subplots(ncols=2, nrows=1)
    _plot(axs[0], h, 'r')
    axs[0].set_title("Original")
    _plot(axs[1], hr, 'g')
    axs[1].set_title("Reconstructed")
    plt.show()


if __name__ == "__main__":
    demo()
