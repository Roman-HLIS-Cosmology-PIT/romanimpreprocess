"""
Helper functions to work with individual bits.

Functions
---------
convert_uint32_to_bits
    Unpack a 2D uint32 cube into a 3D array of bits.

"""

import numpy as np


def convert_uint32_to_bits(arr):
    """
    Converts an (ny,nx) uint32 array into (32,ny,nx) array of 0's and 1's.

    This is mainly so we can visualize a bitmask.

    Parameters
    ----------
    arr : np.array of uint32
        The 2D array to unpack.

    Returns
    -------
    np.array of uint8
        The bit array.

    """

    (ny, nx) = np.shape(arr)
    out = np.zeros((32, ny, nx), dtype=np.uint8)
    for j in range(32):
        out[j, :, :] = ((arr >> j) % 2).astype(np.uint8)
    return out
