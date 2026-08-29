"""Tests for assorted utilities."""

import pytest
from roman_datamodels.dqflags import pixel
from romanimpreprocess.utils.maskhandling import CombinedMask
from romanimpreprocess.utils.visualize import visualize


def test_combinedmask_str():
    """Test that we can input a string to CombinedMask and it gets cast to integer."""

    x = pixel.LOW_QE
    j = 0
    while 2**j < x:
        j += 1

    assert CombinedMask({"LOW_QE": "9"}).array[j] == 9


def test_visualize_err():
    """Tests that visualize raises a ValueError if not enough arguments."""

    with pytest.raises(ValueError):
        visualize(["oops i did it again", "i forgot to provide all the arguments"])
