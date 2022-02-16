import numpy as np
from dataclasses import dataclass
from representations import xy_to_rt, rt_to_xy


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

    def full_period_xy(self, res):
        """
        xy coordinates of a full period of the curve
        :param res: resolution. either number of samples (>1), or the sampling step (<1)
        """
        t = self.full_period_thetas(res)
        r = np.exp(self.log_r(t))
        return rt_to_xy(r, t)

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

    def _plot(h: HuhCurve, *args, **kwargs):
        X = h.full_period_xy(0.1)
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
    import matplotlib.pyplot as plt
    #test()

    h = HuhCurve(HuhParams(m=3, n=2, eps=0.8))

    t = h.full_period_thetas(.01)
    t = t[:int(.3*len(t))]
    r = np.exp(h.log_r(t))

    X = rt_to_xy(r, t)
    rr, tt = xy_to_rt(X)
    XX = rt_to_xy(rr, tt)

    plt.plot(t,'r')
    plt.plot(0,t[0],'ro')
    plt.plot(tt,'b')
    plt.plot(0,tt[0],'bo')
    plt.show()

    _, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
    plt.sca(ax1)
    plt.plot(t, r, 'r')
    i1 = np.argmax(r)
    plt.plot(t[i1], r[i1], 'sr')
    plt.plot(t[0], r[0], 'or')
    plt.plot(tt, rr, 'b')
    i2 = np.argmax(rr)
    plt.plot(tt[i2], rr[i2], 'sb')
    plt.plot(tt[0], rr[0], 'ob')
    plt.sca(ax2)
    plt.plot(X[:, 0], X[:, 1], 'r.')
    plt.plot(X[0, 0], X[0, 1], 'ro')
    plt.plot(X[i1, 0], X[i1, 1], 'rs')
    plt.plot(XX[:, 0], XX[:, 1], 'b.')
    plt.plot(XX[0, 0], XX[0, 1], 'bo')
    plt.plot(XX[i2, 0], XX[i2, 1], 'bs')
    plt.show()
