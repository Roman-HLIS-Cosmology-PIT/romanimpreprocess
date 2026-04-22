"""Test for simple linearity function."""

import numpy as np
from romanimpreprocess.utils.ipc_linearity import _lin_legendre


def test_lin_legendre():
    """Simple test function for _lin_legendre."""
    z = np.linspace(-1.5, 1.5, 31).reshape((1, 31))
    coefs = np.zeros((4, 1, 31))
    coefs[3, :, :] = 1.0
    phi, _ = _lin_legendre(z, coefs)
    print(phi)
    phidiff = phi - np.array(
        [
            -4.0,
            -3.4,
            -2.8,
            -2.2,
            -1.6,
            -1.0,
            -0.4725,
            -0.08,
            0.1925,
            0.36,
            0.4375,
            0.44,
            0.3825,
            0.28,
            0.1475,
            0.0,
            -0.1475,
            -0.28,
            -0.3825,
            -0.44,
            -0.4375,
            -0.36,
            -0.1925,
            0.08,
            0.4725,
            1.0,
            1.6,
            2.2,
            2.8,
            3.4,
            4.0,
        ]
    ).reshape(np.shape(phi))
    assert np.all(phidiff < 1.0e-6)


def test_lin_monomial():
    """Simple test function for _lin_monomial."""
    z = np.linspace(-1.5, 1.5, 31).reshape((1, 31))
    coefs = np.zeros((4, 1, 31))
    coefs[3, :, :] = 1.0
    phi, _ = _lin_legendre(z, coefs)
    print(phi)
    phidiff = phi - z**3
    assert np.all(phidiff < 1.0e-6)
