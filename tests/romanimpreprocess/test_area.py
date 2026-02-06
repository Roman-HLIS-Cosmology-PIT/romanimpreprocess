"""Test for area function."""

import numpy as np
from astropy.wcs import WCS

from romanimpreprocess.utils.coordutils import pixelarea

def test_area_astropy():
    """Test function with astropy WCS."""

    # We'll test both hemispheres.
    for i in range(2):
        w = WCS(naxis=2)
        w.wcs.crpix = [100.5, 100.5]
        w.wcs.cdelt = np.array([-0.1, 0.1])
        w.wcs.crval = [25.0, 83.0 * (1. - 2. * i)]
        w.wcs.ctype = ["RA---STG", "DEC--STG"]

        area = pixelarea(w, N=200)
        s = 0.1 * (np.linspace(0, 199, 200) - 99.5) * np.pi / 180.
        x, y = np.meshgrid(s, s)
        area_target = (0.1 * np.pi / 180.)**2 / (1. + (x**2 + y**2) / 4. )**2
        err = np.log(area / area_target)
        print(err[::199, ::199])
        print(err[99:101, 99:101])
        assert np.all(np.abs(err) < 1.0e-5)
