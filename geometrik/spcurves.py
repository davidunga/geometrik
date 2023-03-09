from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
import numpy as np
from abc import ABC, abstractmethod

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# -------------------


class SmoothParamCurve(ABC):
    """
    Abstract class for smooth parametric curves.
    """

    @abstractmethod
    def __call__(self, t: np.ndarray = None, der: int = 0) -> np.ndarray:
        """
        Get value of curve or its derivatives.
        Args:
            t: curve parameterization. default = native curve parameter.
            der: derivative order.
        Returns:
            ret: np array with shape (len(t), self.ndim)
        """
        pass

    @property
    @abstractmethod
    def t(self) -> np.ndarray:
        """ Get curve parameterization """
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        """ Curve dimensionality """
        pass

    def TN(self, t=None):
        """ Tangent and normal unit vectors """
        assert self.ndim > 1
        dx1 = self(t=t, der=1)
        dx2 = self(t=t, der=2)
        if self.ndim == 2:
            dx1 = np.c_[dx1, np.zeros(len(dx1))]
            dx2 = np.c_[dx2, np.zeros(len(dx2))]
        T = dx1 / np.linalg.norm(dx1, axis=1)[:, None]
        B = np.cross(dx1, dx2)
        B /= np.linalg.norm(B, axis=1)[:, None]
        N = np.cross(B, T)
        return T[:, :self.ndim], N[:, :self.ndim]

    def plot(self, t=None, n_frames=100, axis=None):
        """
        2D plot spline and moving frames
        Args:
            t: parameter values
            n_frames: number of moving frames
            axis: axis to plot - list of 2 integers, default = [0,1]
        """

        X = self(t)
        T, N = self.TN(t)

        if axis is not None:
            axis = [0, 1]

        X = X[:, axis]
        T = T[:, axis]
        N = N[:, axis]

        # trajectory:
        plt.plot(X[:, 0], X[:, 1], 'r-')
        plt.plot(X[0, 0], X[0, 1], 'r*')

        # moving frames:
        ixs = np.round(np.linspace(0, len(T) - 1, min(n_frames, len(X)))).astype(int)
        scale = .025 * np.mean(np.max(X, axis=0) - np.min(X, axis=0))
        for ix in ixs:
            src = X[ix]
            tgt = X[ix] + scale * N[ix]
            plt.plot([src[0], tgt[0]], [src[1], tgt[1]], 'm-')
            tgt = X[ix] + scale * T[ix]
            plt.plot([src[0], tgt[0]], [src[1], tgt[1]], 'c-')


# -------------------


class NDSpline(SmoothParamCurve):
    """
    Spline of multi-dimensional curve.
    """

    def __init__(self, X: np.ndarray, k: int = 4, t=None, w=None, s=0, default_t: np.ndarray = None):
        """

        Args:
            X: N*D array, representing a D-dimensional array
            k: spline order
            t: parameterization.
                Either a 1d array same length as X, or:
                "index" = use index as parameter (default)
                "arclen" = use Euclidean arc length as parameter
                "narclen" = use normalized Euclidean arc length as parameter
            w: smoothing weights.
            s: smoothing parameter.
        """

        if t is None or isinstance(t, str):
            if t is None or t == "index":
                t = np.arange(len(X))
            elif t in ("arclen", "narclen"):
                arclen = np.r_[[0], np.cumsum(np.sqrt(np.diff(X, axis=0) ** 2).sum(axis=1))]
                if t == "arclen":
                    t = arclen
                else:
                    t = arclen / arclen[-1]
            else:
                raise ValueError("Unknown parameterization " + t)

        self._tck, self._u = splprep([X[:, j] for j in range(X.shape[1])], w=w, k=k, s=s, u=t)
        self._knots = np.stack(splev(self._tck[0], self._tck), axis=1)

        if default_t is not None:
            assert np.all(np.diff(default_t) > 0)
            assert default_t[0] >= self._u[0] and default_t[-1] <= self._u[-1]

        self._t = default_t

    def __call__(self, t=None, der=0):
        return np.stack(splev(t if t is not None else self.t, self._tck, der=der), axis=1)

    @property
    def t(self):
        if self._t is None:
            return self._u
        return self._t

    @t.setter
    def t(self, t):
        if t is not None:
            assert np.all(np.diff(t) > 0)
            assert t[0] >= self._u[0] and t[-1] <= self._u[-1]
        self._t = t

    @property
    def ndim(self):
        return len(self._tck[1])


# ----------------------------


class NumericCurve(SmoothParamCurve):

    def __init__(self, X, t):
        self._X = X
        self._t = t

    def __call__(self, t=None, der=0):

        def _interp(X):
            if t is None:
                return X
            return interp1d(self._t, X, axis=0)(t)

        if der == 0:
            return _interp(self._X)

        X = self._X.copy()
        for _ in range(der):
            for j in range(X.shape[1]):
                X[:, j] = np.gradient(X[:, j], self._t, edge_order=1)

        return _interp(X)

    @property
    def t(self):
        return self._t

    @property
    def ndim(self) -> int:
        return self._X.shape[1]
