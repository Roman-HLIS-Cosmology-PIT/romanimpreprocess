"""Test for simple linearity function."""

import numpy as np

from romanimpreprocess.utils.ipc_linearity import _lin


def test_lin():
    """Simple test function for _lin."""
    z = np.linspace(-1.5, 1.5, 31).reshape((1, 31))
    coefs = np.zeros((4, 1, 31))
    coefs[3, :, :] = 1.0
    phi, _ = _lin(z, coefs)
    print(phi)
    assert np.all(phi < -1.0e6)
