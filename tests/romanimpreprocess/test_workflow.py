"""Sim -> L1 -> L2 workflow test."""


import contextlib
import os

import asdf
import numpy as np
from astropy.io import fits
from romanimpreprocess.from_sim import sim_to_isim
from romanimpreprocess.L1_to_L2 import gen_cal_image, gen_noise_image
from romanimpreprocess.utils import ipc_linearity, maskhandling

# Settings for this test
id = 163
sca = 4
tag = "TESTONLY"
dt = 3.04

# This is a shortened 7-read sequence that we don't plan on using in flight.
# I chose it here because it is short and so good for a unit test
# that we want to complete quickly.
#   -- C.H.
#
read_pattern = [[0], [1], [2, 3, 4], [5], [6]]


def genfile(out, v=1):
    """
    Generates a simple FITS image to test calibration.

    Parameters
    ----------
    out : str
        Output FITS file.
    v : int, optional
        Which version of the file to make.

    Returns
    -------
    None

    """

    # make the image
    N = 4088
    img = np.zeros((N, N))
    x_, y_ = np.meshgrid(np.arange(N), np.arange(N))
    for j in range(25):
        x = 10 + (N - 20) * j / 25.0
        y = 10 + (N - 20) * ((13 * j) % 25) / 25.0
        img += 10000 * j * np.exp(-0.5 * ((x_ - x) ** 2 + (y_ - y) ** 2) / 2**2)

    # make FITS header
    phdu = fits.PrimaryHDU(img)
    phdu.header["EXPTIME"] = 139.8
    phdu.header["FILTER"] = "F184"
    phdu.header["CRPIX1"] = (N + 1) / 2.0
    phdu.header["CRPIX2"] = (N + 1) / 2.0
    phdu.header["CD1_1"] = 3.0555555555555554e-05
    phdu.header["CD1_2"] = 0.0
    phdu.header["CD2_1"] = 0.0
    phdu.header["CD2_2"] = 3.0555555555555554e-05
    phdu.header["CTYPE1"] = "RA---TAN-SIP"
    phdu.header["CTYPE2"] = "DEC--TAN-SIP"
    phdu.header["CRVAL1"] = 37.0
    phdu.header["CRVAL2"] = -20.0
    phdu.header["LONPOLE"] = 215.0

    # SIP coefficients
    phdu.header["A_ORDER"] = 2
    phdu.header["A_0_2"] = 2.0e-6
    phdu.header["A_1_1"] = -1.0e-6
    phdu.header["A_2_0"] = 3.0e-6
    phdu.header["B_ORDER"] = 2
    phdu.header["B_0_2"] = 1.4e-5
    phdu.header["B_1_1"] = -1.0e-5
    phdu.header["B_2_0"] = 3.0e-7

    # these coordinates aren't consistent with the SCA location, but shoudn't be a problem for this test.
    phdu.header["RA_TARG"] = 37.0
    phdu.header["DEC_TARG"] = -20.0
    phdu.header["PA_OBSY"] = 185.0

    phdu.writeto(out, overwrite=True)


def _trim(arr, d):
    """
    Trims a 2D array by setting the outer regions to zero.

    Parameters
    ----------
    arr : np.array
        The array (trimmed in place).
    d : int
        How much to trim.

    Returns
    -------
    None

    """

    if d == 0:
        return
    arr[:, :d] = 0
    arr[:, -d:] = 0
    arr[:d, :] = 0
    arr[-d:, :] = 0


def gencal(cstem, rng):
    """
    Writes a bunch of dummy calibration files.

    Parameters
    ----------
    cstem : str
        Stem for the calibration file names.
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    dict
        Dictionary of the calibration files.

    """

    N = 4096  # full size of SCA
    N_ = 4088  # excluding reference pixels
    dtrim = (N - N_) // 2
    x, y = np.meshgrid(np.arange(N), np.arange(N))  # pixel positions, for making maps
    ngrp = len(read_pattern)
    t = np.zeros((ngrp,))
    for j in range(ngrp):
        t[j] = dt * np.mean(np.array(read_pattern[j]))
    print("t =", t)

    ### BIASCORR

    # this is trivial right now, just to check schemas
    asdf.AsdfFile(
        {
            "roman": {
                "data": np.zeros((ngrp, N_, N_), dtype=np.float32),
                "t0": np.mean(np.array(t[1])),  # in seconds
            }
        }
    ).write_to(cstem + f"_biascorr_{tag:s}_SCA{sca:02d}.asdf")

    ### DARK

    dark_slope = 0.005 * 10.0 ** rng.normal(loc=0.0, scale=1.0, size=(N, N))
    _trim(dark_slope, dtrim)
    bias = np.zeros((N, N))
    bias += 13000 + 200 * np.cos(2.0 * np.pi * x / 256.0) + 100 * np.sin(2.0 * np.pi * y / 256) ** 3
    asdf.AsdfFile(
        {
            "roman": {
                "data": np.clip(
                    (bias[None, :, :] + dark_slope[None, :, :] * t[:, None, None]).astype(np.float32),
                    0.0,
                    65535.0,
                ),
                "dq": np.zeros((N, N), dtype=np.uint32),
                "dark_slope": dark_slope.astype(np.float32),
                "dark_slope_err": np.zeros((N, N), dtype=np.float32),
            }
        }
    ).write_to(cstem + f"_dark_{tag:s}_SCA{sca:02d}.asdf")

    ### GAIN

    asdf.AsdfFile(
        {
            "roman": {
                "data": np.clip(1.5 + 0.03 * rng.normal(loc=0.0, scale=1.0, size=(N, N)), 1.4, 1.6),
                "dq": np.zeros((N, N), dtype=np.uint32),
            }
        }
    ).write_to(cstem + f"_gain_{tag:s}_SCA{sca:02d}.asdf")

    ### IPC4D

    K = np.zeros((3, 3, N_, N_), dtype=np.float32)

    # simple IPC kernel
    K[0, 1, :, :] = K[2, 1, :, :] = 0.015  # horizontal
    K[1, 0, :, :] = K[1, 2, :, :] = 0.013  # vertical
    K[0, 0, :, :] = K[2, 2, :, :] = K[0, 2, :, :] = K[2, 0, :, :] = 0.002  # diagonal

    # eliminate dependence on the edges
    K[0, :, 0, :] = 0.0
    K[:, 0, :, 0] = 0.0
    K[-1, :, -1, :] = 0.0
    K[:, -1, :, -1] = 0.0
    K[1, 1, :, :] = 1.0 - np.sum(K, axis=(0, 1))  # normalize to 1

    # write
    asdf.AsdfFile({"roman": {"data": K, "dq": np.zeros((N, N), dtype=np.uint32)}}).write_to(
        cstem + f"_ipc4d_{tag:s}_SCA{sca:02d}.asdf"
    )

    ### LINEARITYLEGENDRE

    # general information
    Smin = 5000 + 500 * np.cos((x + 3 * y) / 100.0)
    Smax = 56000 + 10000 * rng.uniform(size=(N, N))
    Smin = np.clip(Smin, 0.5, 65534.5).astype(np.float32)
    Smax = np.clip(Smax, 0.5, 65534.5).astype(np.float32)
    Sref = (Smin + 300 + 100 * (x % 2)).astype(np.float32)
    pflat = (0.95 + 0.1 * (x / N - 1) - 0.2 * (y / N * (1 - y / N))).astype(np.float32)
    pflat[:dtrim, :] = 0.0
    pflat[-dtrim:, :] = 0.0
    pflat[:, :dtrim] = 0.0
    pflat[:, -dtrim:] = 0.0

    # now build a non-linearity table
    data = np.zeros((3, N, N), dtype=np.float32)
    data[2, :, :] = 20 + 180 * rng.uniform(size=(N, N))

    # now the derivative at Sref is given by:
    # z = 2*(Sref-Smin)/(Smax-Smin)-1
    # deriv = (data[1,:,:] + 3 * data[2,:,:] * z ) * 2/(Smax-Smin)
    # so to make this equal to 1:
    z = 2 * (Sref - Smin) / (Smax - Smin) - 1
    data[1, :, :] = (Smax - Smin) / 2.0 - 3 * data[2, :, :] * z
    # and now the value at reference is zero
    data[0, :, :] = -data[1, :, :] * z - data[2, :, :] * (1.5 * z**2 - 0.5)

    # write
    asdf.AsdfFile(
        {
            "roman": {
                "data": data,
                "dq": np.zeros((N, N), dtype=np.uint32),
                "Smin": Smin,
                "Smax": Smax,
                "Sref": Sref,
                "dark": dark_slope.astype(np.float32),
                "pflat": pflat,
                "ramperr": np.ones((2, N, N), dtype=np.uint16),
            }
        }
    ).write_to(cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")

    # check the mapping
    Slin0, _ = ipc_linearity.linearity(Sref, cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")
    Slinp, _ = ipc_linearity.linearity(Sref + 5, cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")
    Slinm, _ = ipc_linearity.linearity(Sref - 5, cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")
    Slin_der = (Slinp - Slinm) / 10.0
    del Slinp, Slinm
    print(np.amin(Slin0), np.median(Slin0), np.amax(Slin0))
    print(np.amin(Slin_der), np.median(Slin_der), np.amax(Slin_der))

    assert np.amin(Slin0[dtrim:-dtrim, dtrim:-dtrim]) > -1.5
    assert np.amax(Slin0[dtrim:-dtrim, dtrim:-dtrim]) < 1.5
    assert np.amin(Slin_der[dtrim:-dtrim, dtrim:-dtrim]) > 0.99
    assert np.amax(Slin_der[dtrim:-dtrim, dtrim:-dtrim]) < 1.01

    ### MASK

    mask = np.zeros((N, N), dtype=np.uint32)

    # reference pixels
    mask[:dtrim, :] |= 2**31
    mask[-dtrim:, :] |= 2**31
    mask[:, :dtrim] |= 2**31
    mask[:, -dtrim:] |= 2**31
    # dark map
    mask |= np.where(dark_slope > 0.25, np.where(dark_slope > 12.5, 2**11, 2**12), 0).astype(np.uint32)

    asdf.AsdfFile({"roman": {"data": mask}}).write_to(cstem + f"_mask_{tag:s}_SCA{sca:02d}.asdf")

    ### PFLAT

    asdf.AsdfFile({"roman": {"data": pflat, "dq": np.zeros((N, N), np.uint32)}}).write_to(
        cstem + f"_pflat_{tag:s}_SCA{sca:02d}.asdf"
    )

    ### READ

    # this is just a toy case to test out the code
    medband = np.zeros((4096, 128), dtype=np.float32)
    stdband = np.zeros((4096, 128), dtype=np.float32)
    medband[:, :] += 29000.0
    stdband[:, :] += 4.0
    for i in range(16):
        stdband[256 * i, :] = 5
        medband[256 * i, :] += 30
        medband[256 * i + 1, :] += 15

    amp33info = {"valid": True, "med": medband, "std": stdband, "M_PINK": 0.8, "RU_PINK": 1.0}

    asdf.AsdfFile(
        {
            "roman": {
                "anc": {"U_PINK": 1.0, "C_PINK": 2.5},
                "data": (6.0 + 5.0 * rng.uniform(size=(N, N))).astype(np.float32),
                "resetnoise": (25.0 + 5.0 * rng.uniform(size=(N, N))).astype(np.float32),
                "amp33": amp33info,
            }
        }
    ).write_to(cstem + f"_read_{tag:s}_SCA{sca:02d}.asdf")

    ### SATURATION

    asdf.AsdfFile(
        {
            "roman": {
                "data": np.clip(Smax - 50, 1.5, None).astype(np.uint16),
                "dq": np.zeros((N, N), np.uint32),
            }
        }
    ).write_to(cstem + f"_saturation_{tag:s}_SCA{sca:02d}.asdf")

    return {}


def test_run_all(tmp_path):
    """
    Test function for a small pyimcom run.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.

    Returns
    -------
    None

    """

    tmp_dir = str(tmp_path)  # get string version of directory
    rng = np.random.RandomState(seed=1000)  # old RNG, for compatibility
    band = "F184"

    # subdirectories
    with contextlib.suppress(FileExistsError):
        os.mkdir(tmp_dir + "/IN")
    with contextlib.suppress(FileExistsError):
        os.mkdir(tmp_dir + "/OUT-L1")
    with contextlib.suppress(FileExistsError):
        os.mkdir(tmp_dir + "/OUT-L2")

    genfile(tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits", v=1)
    gencal(tmp_dir + "/roman_wfi", rng)

    ctypes = ["linearitylegendre", "gain", "dark", "read", "ipc4d", "flat", "biascorr", "saturation"]
    caldir = {}
    for ctype in ctypes:
        ctype2 = ctype
        if ctype == "flat":
            ctype2 = "pflat"
        caldir[ctype] = tmp_dir + f"/roman_wfi_{ctype2:s}_{tag:s}_SCA{sca:02d}.asdf"
    print(caldir)

    these_reads = []
    for i in range(len(read_pattern)):
        these_reads.append(read_pattern[i][0])
        these_reads.append(read_pattern[i][-1] + 1)
    print("these_reads -->", these_reads)

    sim_to_isim.run_config(
        {
            "IN": tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits",
            "OUT": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
            "READS": these_reads,
            "FITSOUT": True,
            "CALDIR": caldir,
            "CNORM": 1.0,
            "SEED": 100,
        }
    )

    # Below here is stuff for Level 1-->2

    c2 = {
        "IN": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
        "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}.asdf",
        "FITSWCS": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}_asdf_wcshead.txt",
        "CALDIR": caldir,
        "RAMP_OPT_PARS": {"slope": 0.4, "gain": 1.8, "sigma_read": 7.0},
        "JUMP_DETECT_PARS": {"SthreshA": 5.5, "SthreshB": 4.5, "IthreshA": 0.6, "IthreshB": 600.0},
        "SKYORDER": 2,
        "FITSOUT": True,
        "NOISE": {
            "LAYER": ["Rz4S2C1"],
            "TEMP": tmp_dir + f"/temp_{band:s}_{id:d}_{sca:d}.asdf",
            "SEED": 10000,
            "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise.asdf",
        },
    }
    gen_cal_image.calibrateimage(c2 | {"SLICEOUT": True})
    gen_noise_image.generate_all_noise(c2)
    print("\nwrite mask")
    maskhandling.PixelMask1.convert_file(c2["OUT"], c2["OUT"][:-5] + "_mask.fits")

    ### TESTS ON THE OUTPUTS

    dtrim = 4

    # gain map
    with asdf.open(caldir["gain"]) as g_:
        g = np.copy(g_["roman"]["data"])
    print("gain =", np.shape(g))

    # expected signal, in DN/s
    with fits.open(tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits") as f:
        expected_signal = f[0].data[::-1, :] / g[dtrim:-dtrim, dtrim:-dtrim]
        expected_signal /= f[0].header["EXPTIME"]

    with asdf.open(tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}.asdf") as a:
        print(a.info(max_rows=None))
        for i in range(32):
            count = np.count_nonzero(np.bitwise_and(a["roman"]["dq"] >> i, 1))
            print(f"BIT {i:2d} {count:7d}")
            if i == 2:
                assert count > 10000 and count < 300000
        isGood = np.where(a["roman"]["dq"] == 0, 1, 0)

        # now pull out the data, in DN/s
        data_out = np.copy(a["roman"]["data"])

        # sky checks
        print(np.array(a["processinfo"]["skycoefs"]))
        assert len(a["processinfo"]["skycoefs"]) == 6
        assert a["processinfo"]["skycoefs"][0] > 0.5
        assert a["processinfo"]["skycoefs"][0] < 3.0
        for i in range(1, 6):
            assert np.abs(a["processinfo"]["skycoefs"][i]) < 0.6
        skycoefs = np.array(a["processinfo"]["skycoefs"])
        skyresid = np.array(a["roman"]["data_withsky"]) - np.array(a["roman"]["data"])
        # now build residuals from sky model
        N_ = np.shape(skyresid)[-1]
        print("shape", N_)
        u_ = np.linspace(-1.0 + 1.0 / N_, 1.0 - 1.0 / N_, N_)
        u, v = np.meshgrid(u_, u_)
        skyresid -= (
            skycoefs[0]
            + skycoefs[1] * v
            + skycoefs[2] * (1.5 * v**2 - 0.5)
            + skycoefs[3] * u
            + skycoefs[4] * u * v
            + skycoefs[5] * (1.5 * u**2 - 0.5)
        )
        print("MAX", np.amax(np.abs(skyresid)))
        assert np.amax(np.abs(skyresid)) < 1e-3

    print(isGood, np.mean(isGood.astype(np.float32)))

    x = np.where(isGood, data_out - expected_signal, 0.0)
    fits.PrimaryHDU(x).writeto(tmp_dir + "/out_diff.fits", overwrite=True)

    # some quality checks on unmasked pixels
    assert np.count_nonzero(np.abs(x) > 100) < 50
    assert np.count_nonzero(np.logical_and(np.abs(x) > 20, expected_signal < 1)) < 50


# if __name__ == "__main__":
#    test_run_all("out") # <-- comment out in final version
