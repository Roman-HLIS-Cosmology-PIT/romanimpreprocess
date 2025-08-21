"""This package has a bunch of helper functions to work with individual bits.
"""

import numpy as np


def convert_uint32_to_bits(arr):
    """converts an (ny,nx) uint32 array into (32,ny,nx) array of 0's and 1's.
    (mainly so we can visualize a bitmask)
    """
    (ny, nx) = np.shape(arr)
    out = np.zeros((32, ny, nx), dtype=np.uint8)
    for j in range(32):
        out[j, :, :] = ((arr >> j) % 2).astype(np.uint8)
    return out
