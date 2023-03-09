import numpy as np

import geometrik.measures


def get_shape_points(shape, b=None, tnoise=0):

    def _add_t_noise(t):
        return t
        #tt = np.random.rand(len(t))
        #tt = (tt - np.min(tt)) / (np.max(tt) - np.min(tt))
        #return np.sort(tt * (np.max(t) - np.min(t)) + np.min(t))
        #noise = np.mean(np.diff(t)) * tnoise * np.random.randn(len(t))
        #t_ = np.sort(t + noise)
        #t_ = np.min(t_)
        #t_ /= np.max(t_)



    invars = {}
    match shape:

        case 'parabola':
            b = 1 if b is None else b
            t = np.linspace(-1, 1, 1000)
            t = _add_t_noise(t)
            X = np.zeros((len(t), 2), float)
            X[:, 0] = t
            X[:, 1] = b * (t ** 2)
            invars = {"k0": "UNDEF", "k1": 0}

        case 'ellipse':
            a = 1
            b = 2 if b is None else b
            t = np.linspace(0, 2 * np.pi, 1000)
            t = _add_t_noise(t)
            X = np.zeros((len(t), 2), float)
            X[:, 0] = a * np.cos(t)
            X[:, 1] = b * np.sin(t)
            invars = {"k0": 0, "k1": (a * b) ** -(2 / 3)}

        case 'hyperbola':
            a = 1
            b = 2 if b is None else b
            t = np.linspace(1e-1, np.pi - 1e-1, 1000) - .5 * np.pi
            t = _add_t_noise(t)
            X = np.zeros((len(t), 2), float)
            X[:, 0] = a / np.cos(t)
            X[:, 1] = b * np.tan(t)
            invars = {"k0": 0, "k1": -(a * b) ** -(2 / 3)}

        case 'nice_sine':
            # no inflection points
            t = np.linspace(.5, 180 - .5, 1000) * np.pi / 180
            t = _add_t_noise(t)
            X = np.zeros((len(t), 2), float)
            X[:, 0] = t
            X[:, 1] = np.sin(t)

        case 'full_sine':
            t = np.linspace(0, 2 * np.pi, 1000)
            t = _add_t_noise(t)
            X = np.zeros((len(t), 2), float)
            X[:, 0] = t
            X[:, 1] = np.sin(t)

        case _:
            raise ValueError("Unknown shape " + shape)

    return X, t, invars
