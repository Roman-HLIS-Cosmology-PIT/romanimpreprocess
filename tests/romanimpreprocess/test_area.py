
"""Test for area function."""

import numpy as np
from astropy.wcs import WCS
from romanimpreprocess.utils.coordutils import pixelarea


def test_area_astropy():
    """Test function with astropy WCS."""

    # We'll test both hemispheres.
    for i in range(2):
        N = 2000
        d = 0.01
        w = WCS(naxis=2)
        w.wcs.crpix = [N / 2.0 + 0.5, N / 2.0 + 0.5]
        w.wcs.cdelt = np.array([-d, d])
        w.wcs.crval = [25.0, 83.0 * (1.0 - 2.0 * i)]
        w.wcs.ctype = ["RA---STG", "DEC--STG"]

        area = pixelarea(w, N=N)
        s = d * (np.linspace(0, N - 1, N) - N / 2.0 - 0.5) * np.pi / 180.0
        x, y = np.meshgrid(s, s)
        area_target = (d * np.pi / 180.0) ** 2 / (1.0 + (x**2 + y**2) / 4.0) ** 2
        err = np.log(area / area_target)
        print(err)
        print(err[N // 2, N // 2])
        assert np.all(np.abs(err) < 2.0e-4)

    # Test exception handling if we give it the wrong type of object.
    try:
        x = pixelarea("this_isnt_a_wcs_and_should_fail", N=64)
        assert x == 42  # shouldn't get here
    except ValueError as ve:
        assert str(ve) == "Unrecognized WCS type"
