"""
Geometric invariants of smooth parametric curves

Naming convention:
    s = arc-length
    k = curvature
    0, 1, 2 = Full affine, Equi affine, Euclidean
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from geometrik.spcurves import SmoothParamCurve


def geometric_invariants(crv: SmoothParamCurve) -> dict[str, np.ndarray]:
    """
    Euclidean, Equi Affine and Full Affine invariants.
    Returns:
        dict with keys: s0,k0,s1,k1,s2,k2
    """
    s1, k1, s0, k0 = affine_invariants(crv)
    s2, k2 = euclidean_invariants(crv)
    return dict(s0=s0, k0=k0, s1=s1, k1=k1, s2=s2, k2=k2)


def affine_invariants(crv: SmoothParamCurve, sflag: bool = True, kflag: bool = True):
    """
    Full + Equi - Affine curvatures and cumulative arc-lengths
    Args:
        crv: Smooth parametric curve
        sflag: flag - compute arc length
        kflag: flag - compute arc curvature
    Returns:
        s1 : 1d array, same length as crv. Cumulative equi-affine arc length.
        k1 : 1d array, same length as crv. Equi-affine curvature.
        s0 : 1d array, same length as crv. Cumulative full affine arc length. None if sflag = False.
        k0 : 1d array, same length as crv. Full affine curvature. None if kflag = False.
    """

    s1, k1 = equi_affine_invariants(crv)
    s0, k0 = None, None
    if sflag:
        s0 = cumulative_trapezoid(np.sqrt(np.abs(k1)), s1, initial=0)
    if kflag:
        k0 = np.gradient(k1, s1, edge_order=1) * (np.abs(k1) ** -1.5)
    return s1, k1, s0, k0


def equi_affine_invariants(crv: SmoothParamCurve, sflag: bool = True, kflag: bool = True):
    """
    Equi-Affine curvature and cumulative arc-length
    Args:
        crv: Smooth parametric curve
        sflag: Compute arc length
        kflag: Compute curvature
    Returns:
        s1 : 1d array, same length as crv. Cumulative equi-affine arc length. None if sflag = False.
        k1 : 1d array, same length as crv. Equi-affine curvature. None if kflag = False.
    """

    der1, der2 = crv(der=1), crv(der=2)
    d12 = _det(der1, der2)

    s1, k1 = None, None

    if sflag:
        s1 = cumulative_trapezoid(np.abs(d12) ** (1 / 3), crv.t, initial=0)

    if kflag:
        der3 = crv(der=3)
        d13 = _det(der1, der3)
        d23 = _det(der2, der3)
        d14 = _det(der1, crv(der=4))
        d12 = np.cbrt(d12)  # d12 is now actually d12^(1/3)
        a = (4 * d23 + d14) * (d12 ** -5)
        b = (d13 ** 2) * (d12 ** -8)
        k1 = a / 3 - 5 * b / 9

    return s1, k1


def euclidean_invariants(crv: SmoothParamCurve, sflag: bool = True, kflag: bool = True):
    """
    Euclidean curvature and cumulative arc-length
    Args:
        crv: Smooth parametric curve
        sflag: flag - compute arc length
        kflag: flag - compute arc curvature
    Returns:
        s2 : 1d array, same length as crv. Cumulative Euclidean arc length. None if sflag = False.
        k2 : 1d array, same length as crv. Euclidean curvature. None if kflag = False.
    """

    s2, k2 = None, None
    der1 = crv(der=1)
    if sflag:
        s2 = cumulative_trapezoid(np.linalg.norm(der1, axis=1), crv.t, initial=0)
    if kflag:
        k2 = _det(der1, crv(der=2)) / (np.linalg.norm(der1, axis=1) ** 3)
    return s2, k2


def _det(a, b):
    return np.linalg.det(np.stack([a, b], axis=2))
