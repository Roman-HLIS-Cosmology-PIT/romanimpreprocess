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
        w.wcs.crval = [25.0, 83.0 * (1. - 2. * i)]
        w.wcs.ctype = ["RA---STG", "DEC--STG"]

        area = pixelarea(w, N=200)
        s = d * (np.linspace(0, N-1, N) - N / 2.0 - 0.5) * np.pi / 180.
        x, y = np.meshgrid(s, s)
        area_target = (d * np.pi / 180.)**2 / (1. + (x**2 + y**2) / 4. )**2
        err = np.log(area / area_target)
        print(err)
        print(err[N//2, N//2])
        assert np.all(np.abs(err) < 2.0e-4)
