import numpy as np
from scipy.interpolate import interp1d


def randmat(det=None, ortho=False):
    """
    Random 2d square matrix
    :param det: If given, matrix will be scaled to have this determinant
    :param ortho: If True, matrix will be orthogonal
    """
    m = np.random.rand(2, 2)
    if ortho:
        m[:, 1] = [m[1, 0], -m[0, 0]]
        assert np.abs(np.dot(m[:, 0], m[:, 1])) < 1e-6
    if det is not None:
        m *= np.sqrt(np.abs(det / np.linalg.det(m)))
        if np.sign(np.linalg.det(m)) != np.sign(det):
            m = np.fliplr(m)
        assert np.abs(np.linalg.det(m) - det) < 1e-6
    assert np.linalg.matrix_rank(m) == 2
    return m


def extrap_boundaries(y, x=None, b=3):
    """
    Replace values in array y, with an extrapolation
    :param b: boundary size
    """
    x = list(range(len(y))) if x is None else x
    return interp1d(x[b:-b], y[b:-b], fill_value='extrapolate', kind='linear')(x)


def derivative(X, t, n=1):
    """
    n-th order derivatives of X wrt t
    :param X: 2d array, NxD, N = points, D = dim
    :param t: 1d array, parameter to differentiate by.
    :param n: order of derivative
    :return: drvs: list of length n, drvs[i] contains the (i+1)-th order derivative of X
    """
    X = np.expand_dims(X, axis=1) if X.ndim == 1 else X
    drvs = []
    for k in range(n):
        X = np.copy(X)
        for j in range(X.shape[1]):
            X[:, j] = np.gradient(X[:, j], t, edge_order=1)
        drvs.append(X.squeeze())
    return drvs


def winding_angle(X: np.ndarray):
    dX = np.diff(X, axis=0)
    thetas = np.unwrap(np.arctan2(dX[:, 1], dX[:, 0]))
    thetas = np.concatenate([thetas, [2 * thetas[-1] - thetas[-2]]])
    return thetas


def inflection_points(X: np.ndarray):
    """ Index of inflection points in curve """
    ixs = np.nonzero(np.diff(np.sign(np.diff(winding_angle(X)))))[0]
    if len(ixs) > 0:
        ixs += 2
    return ixs


def is_convex(X: np.ndarray):
    return len(inflection_points(X)) == 0


def angdiff(a, b):
    return np.pi - abs(np.mod(abs(a - b), 2 * np.pi) - np.pi)


def fourier(x, t):
    F = np.fft.fft(x)
    n = int(np.ceil(.5 * len(t)) + 1)
    w = 2 * np.pi / (t[-1] - t[0])
    frq = np.arange(0, n - 1) * w
    F = F[:n]
    return F, frq


def _dbg_show_drvs(X, t, n):
    import matplotlib.pyplot as plt
    colors = ('k', 'b', 'r', 'g')
    drvs = [X] + derivative(X, t, n)
    _, axs = plt.subplots(nrows=len(drvs), ncols=X.shape[1])
    for i in range(len(drvs)):
        for j in range(X.shape[1]):
            axs[i, j].plot(drvs[i][:, j], color=colors[i % len(colors)])
            axs[i, j].set_title(f'Order={i} Dim={j}')
    plt.show()


if __name__ == "__main__":
    _dbg_show_drvs(X, t, 3)

