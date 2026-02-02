"""Test for bit-reordering."""

import numpy as np
from romanimpreprocess.utils.bitutils import convert_uint32_to_bits

def test_reorder():
    """Test of bit reordering."""

    im1 = np.zeros((5, 3), dtype=np.uint32)
    im1[4, 2] |= 1 << 7
    im1[2, 1] |= 1 << 12
    im2 = convert_uint32_to_bits(im1)

    assert im2[12, 2, 1] == 1
    assert im2[7, 4, 2] == 1

    # check others are zero
    im2[12, 2, 1] = 0
    im2[7, 4, 2] = 0
    assert np.all(im2 == 0)
