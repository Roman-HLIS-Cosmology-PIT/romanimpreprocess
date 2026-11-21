"""Unit tests for functions in the sky fitting."""

import numpy as np
import pytest
from romanimpreprocess.utils.sky import medfit


def test_errs():
    """Tests for exceptions raised in medfit."""

    with pytest.raises(ValueError):
        medfit(np.zeros((32, 32)), mask=np.zeros((16, 16), dtype=bool))
    with pytest.raises(ValueError):
        medfit(np.zeros((32, 32)), N=2, order=4)
