"""A simple test of sim->L2."""

import asdf
from astropy.io import fits
import numpy as np
from romanimpreprocess.from_sim.sim_to_isim import Image2D, Image2D_from_L1


EXAMPLE_FILE = (
    "https://github.com/Roman-HLIS-Cosmology-PIT/romanimpreprocess/wiki/test-files/"
    "Roman_WAS_truth_H158_887_11.fits.gz"
)


def test_simple(tmp_path):
    """
    This is a simple script to convert Roman to L1/L2.

    For internal testing only, not production.

    Parameters
    ----------
    tmp_path: str-like
        The directory in which to carry out the test.

    Returns
    -------
    None

    """

    tmpdir = str(tmp_path)

    use_read_pattern = [
        [0],
        [1],
        [2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [34],
    ]

    x = Image2D("anlsim", fname=EXAMPLE_FILE)
    print(x.galsimwcs)
    print(x.date, x.idsca)
    print(">>", x.image)
    x.simulate(use_read_pattern)
    x.L1_write_to(tmpdir + "/sim1.asdf")
    x.L2_write_to(tmpdir + "/sim2-direct.asdf")

    with asdf.open("sim1.asdf") as f:
        # print(f.info())
        print("corners:")
        print(f["romanisim"]["wcs"])
        print(f["romanisim"]["wcs"]((0, 0, 4087, 4087), (0, 4087, 0, 4087)))
        print(f["roman"]["meta"])
        fits.PrimaryHDU(f["roman"]["data"]).writeto(tmpdir + "/L1.fits", overwrite=True)

    with Image2D_from_L1(tmpdir + "/sim1.asdf", x.refdata, x.header) as ff:
        ff.pseudocalibrate()
        ff.L2_write_to(tmpdir + "/sim2.asdf")

    with asdf.open(tmpdir + "/sim2.asdf") as f:
        # print(f.info())
        print("corners:")
        print(f["roman"]["meta"]["wcs"]((0, 0, 4087, 4087), (0, 4087, 0, 4087)))
        fits.PrimaryHDU(f["roman"]["data"]).writeto(tmpdir + "/L2.fits", overwrite=True)

    assert use_read_pattern[0][0] == -1  # will fail so we can get the print statements
