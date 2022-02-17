import numpy as np
from dataclasses import dataclass
from representations import CURVE_REP, Curve


@dataclass
class HuhParams:
    # parameters of pure shape.
    m: int          # rotational symmetry (number of max-curvature points)
    n: int          # period (number of rotations of the tangent to the curve)
    eps: float      # amplitude
    t0: float = 0   # phase
    nu: float = 0   # frequency (=m/n)

    def __post_init__(self):
        assert self.n != 0
        self.nu = self.m / self.n


class HuhCurve:

    def __init__(self, params):
        """
        Initialize curve.
        :param params: Either a HuhParams object, or a list of such objects.
        Examples:
            h1 = HuhCurve(HuhParams(m=3, n=2, eps=0.8))
            h2 = HuhCurve([HuhParams(m=6, n=1, eps=1.2), HuhParams(m=2, n=5, eps=0.8)])
        """
        if isinstance(params, HuhParams):
            self.pure_curves = [params]
        else:
            self.pure_curves = list(params)
        assert all([isinstance(c, HuhParams) for c in self.pure_curves])

    def is_pure(self):
        return len(self.pure_curves) == 1

    def period(self):
        """ Get shape's period """
        _nonperiodic_n = 4  # period to use for open shapes
        return 2 * np.pi * np.prod([c.n if c.m != 1 else _nonperiodic_n for c in self.pure_curves])

    def full_period_thetas(self, res):
        """
        Make linearly spaced theta values to cover the full period.
        :param res: resolution. either number of samples (>1), or the sampling step (<1)
        """
        n = res if res > 1 else int(round(self.period() / res))
        return np.linspace(0, self.period(), n)

    def full_period_curve(self, res):
        t = self.full_period_thetas(res)
        r = np.exp(self.log_r(t))
        return Curve((t, r), CURVE_REP.RADIUS_PROFILE)

    def log_r(self, t: np.ndarray):
        """
        log radius of curvature for given thetas
        :param t: 1d array of thetas (winding angles)
        :return: lgr[i] is the log-radius at the i-th point
        """
        lgr = 0
        for c in self.pure_curves:
            lgr += c.eps * np.sin(c.nu * (t - c.t0))
        return lgr

    def __add__(self, other):
        """ Add two shapes """
        return HuhCurve(self.pure_curves + other.pure_curves)

    def __rmul__(self, scale):
        """ Multiply shape by scalar """
        return HuhCurve([HuhParams(m=c.m, n=c.n, eps=scale * c.eps, t0=c.t0)
                         for c in self.pure_curves])


def test():
    import matplotlib.pyplot as plt

    def _plot(h: HuhCurve, *args, **kwargs):
        X = h.full_period_curve(0.1).xy()
        plt.plot(X[:, 0], X[:, 1], *args, **kwargs)
        plt.axis('square')

    h1 = HuhCurve(HuhParams(m=3, n=2, eps=0.8))
    h2 = HuhCurve(HuhParams(m=6, n=1, eps=1.2))
    h3 = h1 + h2

    _, axs = plt.subplots(ncols=3, nrows=1)
    plt.sca(axs[0])
    _plot(h1, 'r')
    plt.sca(axs[1])
    _plot(h2, 'b')
    plt.sca(axs[2])
    _plot(h3, 'g')

    plt.show()


if __name__ == "__main__":
    test()
