import numpy as np


def get_shape_points(shape, b=None):

    if shape == 'parabola':
        b = 0 if b is None else b
        t = np.linspace(-1, 1, 1000)
        X = np.stack([t, t ** 2 + b * t], axis=0).T
        # k1 = 0
        # k0 - undefined.

    elif shape == 'ellipse':
        b = 2 if b is None else b
        t = np.linspace(0, 2 * np.pi, 1000)
        X = np.stack([b * np.cos(t), b * b * np.sin(t)], axis=0).T
        # k1 = b ** -2
        # k0 = 0

    elif shape == 'sine':
        t = np.linspace(0, np.pi, 1000)
        X = np.stack([t, np.sin(t)], axis=0).T

    elif shape == 'full_sine':
        t = np.linspace(0, 2 * np.pi, 1000)
        X = np.stack([t, np.sin(t)], axis=0).T

    elif shape == 'hyperbola':
        b = 2 if b is None else b
        t = np.linspace(1e-1, np.pi - 1e-1, 1000) - .5 * np.pi
        X = np.stack([b * b * np.cos(t) ** -1, - b * np.tan(t)], axis=0).T
        # k1 = -abs(b) ** -2
        # k0 = 0

    else:
        raise ValueError("Unknown shape " + shape)

    return X, t