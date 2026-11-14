"""Test for the image gradient functions."""

import numpy as np
from roman_datamodels.dqflags import pixel
from romanimpreprocess.utils.fitting import build_gradient


def test_build_grad():
    """Test for the build_gradient function."""

    nt, ns = 5, 8  # choose number of slices and array size

    # setup arrays
    data = np.zeros((nt, ns, ns), dtype=np.float32)
    rdq = np.zeros((nt, ns, ns), dtype=np.uint32)
    pdq = np.zeros((nt, ns, ns), dtype=np.uint32)
    tbar = np.linspace(3.0, 15.0, nt)
    rdq[3:, 3, 6] = pixel.SATURATED

    # make a feature somewhere
    data[:, 3:5, 5] = tbar[:, None]
    data[:, 3:6, 0] = 2.0 * tbar[:, None]
    data[1:, 7, 4] = -2.0
    data[1:, 7, 3] = 22.0
    data[2, 3, 6] = 300.0
    data[3:, 3, 6] = 900.0
    data[4, 7, 6] = 90.0

    g = build_gradient(data, rdq, pdq, tbar)

    target = np.zeros((ns, ns), dtype=np.float32)
    target[2, 0] = 1.0
    target[2, 5] = 0.5
    target[2, 6] = 50.0
    target[3, :2] = 1.0
    target[3, 4] = target[3, 6] = 0.5
    target[3, 5] = target[3, 7] = 50.0
    target[4, 1] = 1.0
    target[4, 4:6] = 0.5
    target[4, 6] = 50.0
    target[5, :2] = 1.0
    target[5, 5] = 0.5
    target[6, 0] = 1.0
    assert np.allclose(g, target, atol=0.01)
