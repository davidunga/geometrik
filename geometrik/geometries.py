"""
Supported geometries.
"""

from enum import Enum


class GEOMETRY(Enum):
    EUCLIDEAN = 1
    EQUI_AFFINE = 2
    FULL_AFFINE = 3


GEOMETRIES = [g for g in GEOMETRY]
