"""Tests for various utilities in gen_cal_image."""


from romanimpreprocess.L1_to_L2.gen_cal_image import wcs_from_config


def test_wcs_err():
    """Tests behavior if no WCS."""

    assert wcs_from_config({}) is None
