"""
Geometric invariants of smooth parametric curves

Naming convention:
    s = arc-length
    k = curvature
    0, 1, 2 = Full affine, Equi affine, Euclidean
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from geometrik.spcurves import SmoothParamCurve
from geometrik.spcurve_factory import spatially_separated_points


def geometric_invariants(crv: SmoothParamCurve) -> dict[str, np.ndarray]:
    """
    Euclidean, Equi Affine and Full Affine invariants.
    Returns:
        dict with keys: s0,k0,s1,k1,s2,k2
    """
    s1, k1, s0, k0 = affine_invariants(crv)
    s2, k2 = euclidean_invariants(crv)
    return dict(s0=s0, k0=k0, s1=s1, k1=k1, s2=s2, k2=k2)


def affine_invariants(crv: SmoothParamCurve):

    _eps = 100 * np.finfo(float).eps
    max_invalid_part = .5
    ds1 = 1e-4

    der1, der2 = crv(der=1), crv(der=2)
    d12 = _det(der1, der2)

    s1 = cumulative_trapezoid(np.abs(d12) ** (1 / 3), crv.t, initial=0)

    # ------
    # calc k1:

    ssi = spatially_separated_points(s1, delta=ds1)

    der1 = der1[ssi]
    der2 = der2[ssi]
    der3 = crv(der=3, t=crv.t[ssi])
    der4 = crv(der=4, t=crv.t[ssi])
    d12 = d12[ssi]

    d13 = _det(der1, der3)
    d23 = _det(der2, der3)
    d14 = _det(der1, der4)
    d12 = np.cbrt(d12)  # d12 is now actually d12^(1/3)
    d12_p5 = np.power(d12, 5)
    d12_p8 = np.power(d12, 8)

    ii = (np.abs(d12_p5) > _eps) & (np.abs(d12_p8) > _eps)

    if ii.sum() < len(s1) * (1 - max_invalid_part):
        k1 = np.nan + np.zeros(len(s1))
        s0 = np.nan + np.zeros(len(s1))
        k0 = np.nan + np.zeros(len(s1))
        return s1, k1, s0, k0

    a = (4 * d23[ii] + d14[ii]) / d12_p5[ii]
    b = (d13[ii] ** 2) / d12_p8[ii]

    valid_ixs = np.nonzero(ssi)[0][ii]
    k1 = interp1d(s1[valid_ixs], a / 3 - 5 * b / 9, axis=0, fill_value="extrapolate")(s1)

    # ------
    # got s1 and k1. calc s0 and k0:

    k1_pwr = np.power(np.abs(k1[valid_ixs]), 1.5)
    valid_ixs = valid_ixs[k1_pwr > _eps]

    if len(valid_ixs) < len(s1) * (1 - max_invalid_part):
        s0 = np.nan + np.zeros(len(s1))
        k0 = np.nan + np.zeros(len(s1))
        return s1, k1, s0, k0

    k0_valid = np.gradient(k1[valid_ixs], s1[valid_ixs], edge_order=1) / k1_pwr[k1_pwr > _eps]
    k0 = interp1d(s1[valid_ixs], k0_valid, axis=0, fill_value="extrapolate")(s1)
    s0 = cumulative_trapezoid(np.sqrt(np.abs(k1)), s1, initial=0)

    return s1, k1, s0, k0


def euclidean_invariants(crv: SmoothParamCurve):
    der1 = crv(der=1)
    s2 = cumulative_trapezoid(np.linalg.norm(der1, axis=1), crv.t, initial=0)
    k2 = _det(der1, crv(der=2)) / (np.linalg.norm(der1, axis=1) ** 3)
    return s2, k2


def _det(a, b):
    return np.linalg.det(np.stack([a, b], axis=2))
