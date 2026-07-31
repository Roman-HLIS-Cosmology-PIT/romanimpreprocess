import numpy as np
from romanimpreprocess.utils.orientation import get_orientation, sca_ref_pos


def _wrap_orient(meta):
    """Simple wrapper."""
    return get_orientation({"roman": {"meta": meta}})


def test_get_orientation():
    """Some test cases for the orientation."""

    out1 = _wrap_orient({"wcsinfo": {"dec_ref": 0.0, "ra_ref": 0.0, "roll_ref": 0.0}})
    assert 0.4295 < out1["ra"] < 0.4296
    assert -0.24805 < out1["dec"] < -0.24795
    assert 119.99 < out1["pa"] < 120.01

    # simple rotation should be good here
    xwfi = sca_ref_pos[:, 0]
    ywfi = sca_ref_pos[:, 1] + 0.496
    ra_expect = np.sqrt(0.75) * ywfi + 0.5 * xwfi
    dec_expect = -0.5 * ywfi + np.sqrt(0.75) * xwfi
    assert np.all(np.abs(ra_expect - out1["ra_sca"]) < 1.0e-4)
    assert np.all(np.abs(dec_expect - out1["dec_sca"]) < 1.0e-4)

    # now with declination
    out2 = _wrap_orient({"wcsinfo": {"dec_ref": 0.5, "ra_ref": 0.0, "roll_ref": 0.0}})
    assert 0.4295 < out2["ra"] < 0.4296
    assert 0.25195 < out2["dec"] < 0.25205
    assert 119.99 < out2["pa"] < 120.01

    # simple rotation should be good here
    xwfi = sca_ref_pos[:, 0]
    ywfi = sca_ref_pos[:, 1] + 0.496
    ra_expect = np.sqrt(0.75) * ywfi + 0.5 * xwfi
    dec_expect = -0.5 * ywfi + np.sqrt(0.75) * xwfi + 0.5
    assert np.all(np.abs(ra_expect - out2["ra_sca"]) < 1.0e-4)
    assert np.all(np.abs(dec_expect - out2["dec_sca"]) < 1.0e-4)

    # now with RA
    out3 = _wrap_orient({"wcsinfo": {"dec_ref": 0.5, "ra_ref": 247.0, "roll_ref": 0.0}})
    assert np.abs(out3["ra"] - out2["ra"] - 247.0) < 1.0e-5
    assert np.abs(out3["dec"] - out2["dec"]) < 1.0e-5
    assert np.abs(out3["pa"] - out2["pa"]) < 1.0e-5
    assert np.all(np.abs(out3["ra_sca"] - out2["ra_sca"] - 247.0) < 1.0e-5)
    assert np.all(np.abs(out3["dec_sca"] - out2["dec_sca"]) < 1.0e-5)
    # rollover with RA
    out3 = _wrap_orient({"wcsinfo": {"dec_ref": 0.5, "ra_ref": 359.6, "roll_ref": 0.0}})
    assert np.abs(out3["ra"] - out2["ra"] + 0.4) < 1.0e-5
    assert np.abs(out3["dec"] - out2["dec"]) < 1.0e-5
    assert np.abs(out3["pa"] - out2["pa"]) < 1.0e-5
    assert np.all(
        np.amin(np.abs(out3["ra_sca"] - out2["ra_sca"] - 359.6, out3["ra_sca"] - out2["ra_sca"] + 0.4))
        < 1.0e-5
    )
    assert np.all(np.abs(out3["dec_sca"] - out2["dec_sca"]) < 1.0e-5)

    # roll test
    out4 = _wrap_orient({"wcsinfo": {"dec_ref": 0.0, "ra_ref": 0.0, "roll_ref": 330.0}})
    assert 0.49595 < out4["ra"] < 0.49605
    assert -5.0e-5 < out4["dec"] < 5.0e-5
    assert 89.99 < out4["pa"] < 90.01

    # simple rotation should be good here
    xwfi = sca_ref_pos[:, 0]
    ywfi = sca_ref_pos[:, 1] + 0.496
    ra_expect = ywfi
    dec_expect = xwfi
    assert np.all(np.abs(ra_expect - out4["ra_sca"]) < 1.0e-4)
    assert np.all(np.abs(dec_expect - out4["dec_sca"]) < 1.0e-4)

    # roll test - other way
    out4 = _wrap_orient({"wcsinfo": {"dec_ref": 0.0, "ra_ref": 0.0, "roll_ref": 150.0}})
    assert 0.49595 < 360.0 - out4["ra"] < 0.49605
    assert -5.0e-5 < out4["dec"] < 5.0e-5
    assert 269.99 < out4["pa"] < 270.01

    # simple rotation should be good here
    xwfi = sca_ref_pos[:, 0]
    ywfi = sca_ref_pos[:, 1] + 0.496
    ra_expect = 360.0 - ywfi
    dec_expect = -xwfi
    assert np.all(np.abs(ra_expect - out4["ra_sca"]) < 1.0e-4)
    assert np.all(np.abs(dec_expect - out4["dec_sca"]) < 1.0e-4)
