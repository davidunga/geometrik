"""
Supported geometries.
"""

from enum import Enum


class GEOMETRY(Enum):
    EUCLIDEAN = 1
    EQUI_AFFINE = 2
    FULL_AFFINE = 3


GEOMETRIES = [g for g in GEOMETRY]


def convention2geom(c: int | str) -> GEOMETRY:

    if isinstance(c, str):
        c = int(c)

    match c:
        case 0:
            return GEOMETRY.FULL_AFFINE
        case 1:
            return GEOMETRY.EQUI_AFFINE
        case 2:
            return GEOMETRY.EUCLIDEAN
        case _:
            raise ValueError("Unknown convention")
