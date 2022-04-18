from enum import Enum


class GEOMETRY(Enum):
    FULL_AFFINE = 'full_affine'
    EQUI_AFFINE = 'equi_affine'
    EUCLIDEAN = 'euclidean'


GEOMETRIES = [e for e in GEOMETRY]