"""Sim -> L1 -> L2 workflow test."""


import contextlib
import os

import asdf
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS
from PIL import Image
from roman_datamodels.dqflags import pixel
from romanimpreprocess.from_sim import sim_to_isim
from romanimpreprocess.L1_to_L2 import gen_cal_image, gen_noise_image
from romanimpreprocess.utils import fpaplot, ipc_linearity, maskhandling, orientation, visualize

# Settings for this test
id = 163
sca = 4
tag = "TESTONLY"
dt = 3.04

# This is a shortened 14-read sequence that we don't plan on using in flight.
# I chose it here because it is short and so good for a unit test
# that we want to complete quickly.
#   -- C.H.
#
read_pattern = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10], [11, 12], [13]]


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
    data = np.zeros((4, N, N), dtype=np.float32)
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

    asdf.AsdfFile({"roman": {"dq": mask}}).write_to(cstem + f"_mask_{tag:s}_SCA{sca:02d}.asdf")

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
                "anc": {"U_PINK": 0.4, "C_PINK": 0.8},
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
                "data": np.clip(Smax - 50, 1.5, None).astype(np.float32),
                "dq": np.zeros((N, N), np.uint32),
            }
        }
    ).write_to(cstem + f"_saturation_{tag:s}_SCA{sca:02d}.asdf")

    ### DARKDECAY

    dectab = {}
    for k in range(1, 19):
        dectab[f"WFI{k:02d}"] = {"amplitude": 0.3 + 0.1 * np.cos(k), "time_constant": 20.0 + k}
    asdf.AsdfFile({"roman": {"decay_table": dectab}}).write_to(
        cstem + f"_darkdecay_{tag:s}_SCA{sca:02d}.asdf"
    )

    return {}


def forward_backward_lin_ilin(linearity_file):
    """
    Forward-backward test for linearity routines.

    Parameters
    ----------
    linearity_file : str
        ASDF linearity calibration reference file.

    Returns
    -------
    None

    """

    ymin = 260
    ymax = 262
    xmin = 140
    xmax = 143
    dy = ymax - ymin
    dx = xmax - xmin
    with asdf.open(linearity_file) as F:
        print("Smin", F["roman"]["Smin"][ymin:ymax, xmin:xmax])
        print("Smax", F["roman"]["Smax"][ymin:ymax, xmin:xmax])
        S = F["roman"]["Sref"][ymin:ymax, xmin:xmax]
        S[:, :] += 5000.0 * np.linspace(0, dx * dy - 1, dx * dy).reshape((dy, dx))
    Slin, dq = ipc_linearity.linearity(S, linearity_file, origin=(xmin, ymin))
    Sfwd, exflag = ipc_linearity.invlinearity(Slin, linearity_file, origin=(xmin, ymin))

    print("coefs:")
    with asdf.open(linearity_file) as F:
        print(F["roman"]["data"][:, ymin:ymax, xmin:xmax])
    print("signal [DN_raw]:")
    print(S)
    print("inverted signal [DN_lin]:")
    print(Slin)
    print("recovered signal [DN_raw]:")
    print(Sfwd)
    print("flags")
    print(dq, exflag)

    # was the recovery within bounds?
    assert not np.any(exflag)
    print(Sfwd - S)
    assert np.amax(np.abs(Sfwd - S)) < 0.002


def il_example(linearity_file, gain_file, ipc_file):
    """
    Some tests for the inverse linearity class.

    Parameters
    ----------
    linearity_file : str
        ASDF linearity calibration reference file.
    gain_file : str
        ASDF gain calibration reference file.
    ipc_file : str
        ASDF ipc4d calibration reference file.

    Returns
    -------
    None

    """

    # expected results
    target1 = np.array(
        [[4801.0491668, 4900.74928657, 4800.50198393], [4800.30217909, 4900.15392476, 4800.05504147]]
    )
    target2 = np.array(
        [[4803.76066256, 4920.3284374, 4803.19832938], [4817.8237426, 6177.69747299, 4817.69985963]]
    )
    target3 = np.array([[3.99985202, 30.00056007, 4.00053699], [25.99872292, 1871.82650119, 26.00227536]])

    ILTEST = ipc_linearity.IL(linearity_file, gain_file, ipc_file)
    n = 4088
    NE = np.zeros((n, n), dtype=np.float32)
    ymin = 260
    ymax = 262
    xmin = 140
    xmax = 143
    val1 = ILTEST.apply(NE, electrons=True, electrons_out=False)[ymin:ymax, xmin:xmax]
    print(val1)
    assert np.all(np.abs(target1 - val1) < 0.002)
    NE[::3, ::3] = 2.0e3
    val2 = ILTEST.apply(NE, electrons=True, electrons_out=False)[ymin:ymax, xmin:xmax]
    print(val2)
    assert np.all(np.abs(target2 - val2) < 0.002)
    val3 = ILTEST.apply(NE, electrons=True, electrons_out=True)[ymin:ymax, xmin:xmax]
    print(val3)
    assert np.all(np.abs(target3 - val3) < 0.002)

    assert np.shape(ILTEST.apply(NE, electrons=True, electrons_out=False)) == (4088, 4088)
    assert np.shape(ILTEST.apply(NE, electrons=True, electrons_out=True)) == (4088, 4088)


def run_all(tmp_path, subtr, checkdata, cleanup=True):
    """
    Test function for a small pyimcom run.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.
    subtr : bool
        Test whether to package the reference array?
    checkdata : dict
        Dictionary of properties to check. Includes the keys:

        - ``wfi18p1``: 1st percentile of each time slice of the L1 with WFI18 transient
        - ``wfi18p99``: 99th percentile of each time slice of the L1 with WFI18 transient

    cleanup : bool, optional
        Whether to clean up files when done? (Only turn off if you are running on a local
        machine to look at artifacts.)

    Returns
    -------
    image : np.ndarray
        The slope image from the default run.
    mask : np.ndarray
        The mask from the default run.

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

    # forward-backward test (contains its own asserts)
    forward_backward_lin_ilin(caldir["linearitylegendre"])
    il_example(caldir["linearitylegendre"], caldir["gain"], caldir["ipc4d"])

    # make sample images
    arr = fpaplot.multi_image(
        tmp_dir + "/roman_wfi_{:s}_" + tag + "_SCA{:02d}.asdf", 128, maskhandling.PixelMask1
    )
    # check a few pixels in the output image
    checkpix = np.array(
        [
            [2057, 672, 60, 78, 138],
            [2057, 1527, 175, 220, 46],
            [709, 1624, 68, 1, 84],
            [0, -1, 255, 255, 255],
            [455, 422, 255, 255, 255],
            [455, 424, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    for i in range(np.shape(checkpix)[0]):
        diff = arr[checkpix[i, 0], checkpix[i, 1], :].astype(np.int16) - checkpix[i, -3:]
        assert np.all(np.abs(diff) < 16)
    Image.fromarray(arr[::-1, :, :]).save("panel_image.png")

    supp = {"EXTRACT_REF": {"data_encoding_offset": 4000}} if subtr else {}
    sim_to_isim.run_config(
        {
            "IN": tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits",
            "OUT": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
            "READS": these_reads,
            "FITSOUT": True,
            "CALDIR": caldir,
            "CNORM": 1.0,
            "SEED": 200,
        }
        | supp
    )

    # this will have darkdecay
    sim_to_isim.run_config(
        {
            "IN": tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits",
            "OUT": tmp_dir + f"/OUT-L1/sim_L1withdd_{band:s}_{id:d}_{sca:d}.asdf",
            "READS": these_reads,
            "FITSOUT": True,
            "CALDIR": caldir,
            "CNORM": 1.0,
            "SEED": 200,
        }
        | supp
    )

    # test extraction of the orientation
    ordict = orientation.get_orientation(tmp_dir + f"/OUT-L1/sim_L1withdd_{band:s}_{id:d}_{sca:d}.asdf")
    print(ordict)
    assert 37.045 < ordict["ra"] < 37.047
    assert -19.507 < ordict["dec"] < -19.505
    assert 4.999 < ordict["pa"] < 5.001
    assert len(ordict["ra_sca"]) == 18
    assert len(ordict["dec_sca"]) == 18

    # copy this file into a place for WFI18 (so that we can test transient function)
    with asdf.open(tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf") as a:
        a["roman"]["meta"]["instrument"]["detector"] = "WFI18"
        # this is just a toy example to exercise that the WFI18 correction is "on"
        newdata = a["roman"]["data"][0, 4:-4, 4:-4].astype(np.float32)
        rows = np.linspace(4, 4091, 4088)
        rows += rows // 256 * 4  # insert a timing gap
        newdata += (-80.0 * np.exp(-rows / 150.0) + 5.0 * np.exp(-rows / 1300.0)).astype(np.float32)[:, None]
        a["roman"]["data"][0, 4:-4, 4:-4] = np.clip(newdata, 0, 65535)
        a.write_to(tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_18.asdf")

    # Below here is stuff for Level 1-->2

    c2 = {
        "IN": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
        "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}.asdf",
        "FITSWCS": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}_asdf_wcshead.txt",
        "CALDIR": caldir,
        "RAMP_OPT_PARS": {"slope": 0.4, "gain": 1.8, "sigma_read": 7.0},
        "JUMP_DETECT_PARS": {"SthreshA": 10.0, "SthreshB": 4.5, "IthreshA": 0.6, "IthreshB": 600.0},
        "SKYORDER": 2,
        "FITSOUT": True,
        "NOISE": {
            "LAYER": ["Rz4S2C1", "O", "Prb2"],
            "TEMP": tmp_dir + f"/temp_{band:s}_{id:d}_{sca:d}.asdf",
            "SEED": 10000,
            "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise.asdf",
        },
    }
    if subtr:
        c2["EXCLUDE_FIRST"] = False
    gen_cal_image.calibrateimage(c2 | {"SLICEOUT": True})
    gen_noise_image.generate_all_noise(c2)
    print("\nwrite mask")
    maskhandling.PixelMask1.convert_file(c2["OUT"], c2["OUT"][:-5] + "_mask.fits")

    # Also exercise the romancal likelihood ramp-fit path
    # Turn on the WFI18 transient step, which is a no-op here because this is WFI04.
    c_rc = c2 | {
        "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_romancalfit.asdf",
        "romancal_ramp_fit": True,
        "correct_wfi18_transient": True,
    }
    gen_cal_image.calibrateimage(c_rc)

    # now do a test as if this were WFI18 to exercise the post-reset transient functionality
    c2x = c2 | {"correct_wfi18_transient": True, "EXCLUDE_FIRST": False}
    c2x["IN"] = tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_18.asdf"
    c2x["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_18_noexcludefirst.asdf"
    gen_cal_image.calibrateimage(c2x)
    c4 = c2 | {"correct_wfi18_transient": True, "EXCLUDE_FIRST": False}
    c4["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noexcludefirst.asdf"
    gen_cal_image.calibrateimage(c4)

    # this one tests dark decay
    # right now not actually implemented -- this shows up as a change in the "sky"
    c2d = c2.copy()
    c2d["CALDIR"] = c2["CALDIR"] | {"dark_decay": tmp_dir + f"/roman_wfi_darkdecay_{tag:s}_SCA{sca:02d}.asdf"}
    c2d["IN"] = tmp_dir + f"/OUT-L1/sim_L1withdd_{band:s}_{id:d}_{sca:d}.asdf"
    c2d["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_withdecay.asdf"
    gen_cal_image.calibrateimage(c2d)

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
        cry, crx = np.where(np.bitwise_and(a["roman"]["dq"] >> 2, 1))
        print("FIRST CRs:")
        print(cry[:10])
        print(crx[:10])
        v = max(cry[0], 3)
        u = max(crx[0], 3)
        print("2D image -->")
        print(a["roman"]["data"][v - 3 : v + 3, u - 3 : u + 3])

        for i in range(32):
            count = np.count_nonzero(np.bitwise_and(a["roman"]["dq"] >> i, 1))
            print(f"BIT {i:2d} {count:7d}")
            if i == 2:
                assert count > 10000 and count < 30000
        isGood = np.where(a["roman"]["dq"] == 0, 1, 0)
        dq_local = np.asarray(a["roman"]["dq"])
        err_local = np.asarray(a["roman"]["err"], dtype=np.float64)
        vp_local = np.asarray(a["roman"]["var_poisson"], dtype=np.float64)

        # now pull out the data, in DN/s
        data_out = np.copy(a["roman"]["data"])

        # sky checks
        print(np.array(a["processinfo"]["skycoefs"]))
        assert len(a["processinfo"]["skycoefs"]) == 6
        assert -0.3 <= a["processinfo"]["skycoefs"][0] <= 1.7
        for i in range(1, 6):
            assert np.abs(a["processinfo"]["skycoefs"][i]) < 1.0
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

        print(a["processinfo"]["romanimpreprocess_env"])
        assert "." in a["processinfo"]["romanimpreprocess_env"]["romanimpreprocess"]

    hisignal = np.logical_and(isGood, expected_signal > 5.0)
    print(isGood, np.mean(isGood.astype(np.float32)), "hisig =", np.count_nonzero(hisignal))

    x = np.where(isGood, data_out - expected_signal, 0.0)
    # fits.PrimaryHDU(x).writeto(tmp_dir + "/out_diff.fits", overwrite=True)

    # some quality checks on unmasked pixels
    assert np.count_nonzero(np.abs(x) > 100) < 50
    assert np.count_nonzero(np.logical_and(np.abs(x) > 20, expected_signal < 1)) < 50

    # checks on the romancal likelihood ramp-fit output
    with asdf.open(c_rc["OUT"]) as a_rc:
        assert np.shape(a_rc["roman"]["data"]) == (4088, 4088)
        assert "dumo" in a_rc["roman"]
        assert "chisq" in a_rc["roman"]
        isGood_rc = a_rc["roman"]["dq"] == 0
        data_rc = np.asarray(a_rc["roman"]["data"])
        assert np.all(np.isfinite(data_rc[isGood_rc]))
        x_rc = np.where(isGood_rc, data_rc - expected_signal, 0.0)
        assert np.count_nonzero(np.abs(x_rc) > 100) < 50

        # The local and romancal-likelihood fitters use the same data, so on
        # common good pixels they should report similar uncertainties, agree well
        # within those uncertainties, and flag a similar number of cosmic rays.
        dq_rc = np.asarray(a_rc["roman"]["dq"])
        err_rc = np.asarray(a_rc["roman"]["err"], dtype=np.float64)
        vp_rc = np.asarray(a_rc["roman"]["var_poisson"], dtype=np.float64)
        common = isGood.astype(bool) & isGood_rc

        # median uncertainties shouldn't change by more than ~5%
        for name, m_loc, m_rc in [
            ("err", err_local, err_rc),
            ("var_poisson", vp_local, vp_rc),
        ]:
            ratio = np.median(m_rc[common]) / np.median(m_loc[common])
            print(f"uncertainty ratio rc/local {name:>11} = {ratio:.3f}")
            assert 0.95 < ratio < 1.05

        # the two fits should agree well within their reported uncertainty
        z = mad_std(((data_out - data_rc) / err_rc)[common])
        # cosmic-ray flagging should be similar
        jump_local = np.count_nonzero((dq_local & pixel.JUMP_DET) != 0)
        jump_rc = np.count_nonzero((dq_rc & pixel.JUMP_DET) != 0)
        print(f"mad_std((local-rc)/err)={z:.4f}, jump_local={jump_local}, jump_rc={jump_rc}")
        assert z < 0.05
        assert 100 < jump_rc < 2 * jump_local

    # checks on the WFI18 version
    with asdf.open(c2x["IN"]) as _a:
        assert np.allclose(
            np.percentile(_a["roman"]["data"], 99, axis=(-2, -1)), checkdata["wfi18p99"], atol=2.2, rtol=0.0
        )
        assert np.allclose(
            np.percentile(_a["roman"]["data"], 1, axis=(-2, -1)), checkdata["wfi18p1"], atol=2.2, rtol=0.0
        )

    with asdf.open(c4["OUT"]) as a_notr, asdf.open(c2x["OUT"]) as a_withtr:
        diff = a_withtr["roman"]["data"] - a_notr["roman"]["data"]
        # the 10/20/80/90th percentile wthiout correction are -0.078/-0.049/+0.029/+0.031
        assert np.percentile(diff, 10) > -0.014
        assert np.percentile(diff, 20) > -0.007
        assert np.percentile(diff, 80) < 0.007
        assert np.percentile(diff, 90) < 0.014
        # make sure they are different!
        assert np.percentile(diff, 80) - np.percentile(diff, 20) > 1e-4

    # checks on the darkdecay version
    with asdf.open(c2["OUT"]) as a_orig, asdf.open(c2d["OUT"]) as a_new:
        diff = a_new["roman"]["data"] - a_orig["roman"]["data"]
        diff1d = np.median(diff, axis=1)
        assert np.all(np.abs(diff1d) < 1.0e-4)

        # sky goes "up" because we are correcting a downward slope
        skydiff = a_new["processinfo"]["skycoefs"] - a_orig["processinfo"]["skycoefs"]
        assert 0.004 < skydiff[0] < 0.007
        assert np.all(np.abs(skydiff[1:]) < 0.0015)

    # check that we can convert the output to PDF
    visualize.visualize(
        [
            None,
            tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
            "128,256,512,640",
            tmp_dir + "/out_im1.pdf",
            0.5,
        ]
    )
    assert os.path.exists(tmp_dir + "/out_im1.pdf")

    # noise tests
    with asdf.open(tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise.asdf") as a:
        print(a.info(max_rows=None))
        adata = a["noise"]
        nlayer = len(c2["NOISE"]["LAYER"])
        assert np.shape(adata) == (nlayer, 4088, 4088)
        for j in range(nlayer):
            print("layer", j, c2["NOISE"]["LAYER"][j])
            x = np.where(isGood, adata[j, :, :], 0.0)
            print(np.percentile(x, 0.1), np.percentile(x, 5), np.percentile(x, 95), np.percentile(x, 99.9))
            x2 = adata[j, :, :][hisignal]
            print(np.percentile(x2, 25), np.percentile(x2, 75))

            # assertions
            if j == 0:
                assert 0.7 < np.percentile(x, 95) - np.percentile(x, 5) < 1.1
                assert 0.3 < np.percentile(x2, 75) - np.percentile(x2, 25) < 0.4
            if j == 1:
                assert 0.14 < np.percentile(x, 95) - np.percentile(x, 5) < 0.40
                assert 1.1 < np.percentile(x2, 75) - np.percentile(x2, 25) < 1.4
            if j == 2:
                assert 0.14 < np.percentile(x, 95) - np.percentile(x, 5) < 0.40

    # test truncation to float16 on a clipped layer
    c3 = c2 | {"NOISE_PRECISION": 16}
    c3["NOISE"]["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise16.asdf"
    gen_noise_image.generate_all_noise(c3)
    p = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}"
    with asdf.open(p + "_noise.asdf") as a_orig, asdf.open(p + "_noise16.asdf") as a16:
        diff = (a16["noise"][0, :, :] - a_orig["noise"][0, :, :]) / (1.0 + np.abs(a_orig["noise"][0, :, :]))
    assert np.all(np.abs(diff) < 0.005)

    # this should raise an exception, we will check that
    c3["NOISE_PRECISION"] = -1
    try:
        gen_noise_image.generate_all_noise(c3)
        assert c3["NOISE_PRECISION"] == 0  # shouldn't get here
    except ValueError as ve:
        assert str(ve) == "Unsupported noise precision."

    with fits.open(str(tmp_path) + "/OUT-L2/sim_L2_F184_163_4_asdf_to.fits") as f:
        image = np.copy(f[0].data)
        mask = np.copy(f[1].data)

    if cleanup:
        for rf in [
            "out_im1.pdf",
            "roman_wfi_linearitylegendre_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_L2.asdf",
            "roman_wfi_biascorr_TESTONLY_SCA04.asdf",
            "roman_wfi_mask_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_refL2_asdf_to.fits",
            "roman_wfi_dark_TESTONLY_SCA04.asdf",
            "roman_wfi_pflat_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_refL2.asdf",
            "roman_wfi_darkdecay_TESTONLY_SCA04.asdf",
            "roman_wfi_read_TESTONLY_SCA04.asdf",
            "temp_F184_163_4.asdf",
            "roman_wfi_gain_TESTONLY_SCA04.asdf",
            "roman_wfi_saturation_TESTONLY_SCA04.asdf",
            "roman_wfi_ipc4d_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_L2_asdf_to.fits",
        ]:
            os.remove(tmp_path / rf)
        for rf in ["Roman_Test_truth_F184_163_4.fits"]:
            os.remove(tmp_path / f"IN/{rf}")
        for rf in [
            "sim_L1_F184_163_18.asdf",
            "sim_L1_F184_163_4.asdf",
            "sim_L1withdd_F184_163_4.asdf",
            "sim_L1_F184_163_4_asdf_to.fits",
            "sim_L1withdd_F184_163_4_asdf_to.fits",
            "sim_L1_F184_163_4_asdf_wcshead.txt",
            "sim_L1withdd_F184_163_4_asdf_wcshead.txt",
        ]:
            os.remove(tmp_path / f"OUT-L1/{rf}")
        for rf in [
            "sim_L2_F184_163_18_noexcludefirst_asdf_to.fits",
            "sim_L2_F184_163_4_noexcludefirst.asdf",
            "sim_L2_F184_163_4_romancalfit_asdf_to.fits",
            "sim_L2_F184_163_18_noexcludefirst.asdf",
            "sim_L2_F184_163_4_noise_asdf_to.fits",
            "sim_L2_F184_163_4_romancalfit.asdf",
            "sim_L2_F184_163_4_asdf_to.fits",
            "sim_L2_F184_163_4_noise.asdf",
            "sim_L2_F184_163_4_withdecay_asdf_to.fits",
            "sim_L2_F184_163_4_mask.fits",
            "sim_L2_F184_163_4_noise16_asdf_to.fits",
            "sim_L2_F184_163_4_withdecay.asdf",
            "sim_L2_F184_163_4_noexcludefirst_asdf_to.fits",
            "sim_L2_F184_163_4_noise16.asdf",
            "sim_L2_F184_163_4.asdf",
        ]:
            os.remove(tmp_path / f"OUT-L2/{rf}")

    return image, mask


def test_run_all(tmp_path):
    """Main workflow test."""

    # Test with moving the reference.
    im0, mask0 = run_all(
        tmp_path,
        True,
        {
            "wfi18p99": np.array([4030.0, 4034.0, 4044.0, 4060.0, 4068.0]),
            "wfi18p1": np.array([3946.0, 3979.0, 3984.0, 3984.0, 3982.0]),
        },
    )

    # Test without moving the reference.
    im1, mask1 = run_all(
        tmp_path,
        False,
        {
            "wfi18p99": np.array([5925.0, 5927.0, 5931.0, 5937.0, 5944.0, 5947.0]),
            "wfi18p1": np.array([4780.0, 4785.0, 4788.0, 4793.0, 4796.0, 4797.0]),
        },
    )

    # Check that we got the same answer.
    thresh = 2  # want to allow the occasional floating point moves over a flag boundary.
    assert np.count_nonzero(mask0 != mask1) <= thresh
    err = np.abs(im1 - im0) / (1.0 + np.abs(im1))
    assert np.count_nonzero(err > 3.0e-4) <= thresh


def test_flip(tmp_path):
    """Test of flipping the SCA."""

    # Build the data.
    # Note this makes a header with SIP coefficients.
    tmpdir = str(tmp_path)
    genfile(tmpdir + "/test1.fits", v=1)
    with fits.open(tmpdir + "/test1.fits", memmap=False, lazy_load_hdus=False) as f:
        my_hdu = fits.PrimaryHDU(f[0].data, header=f[0].header.copy())

    # Now the flipped image
    sim_to_isim.hdu_sip_hflip(my_hdu.data, my_hdu.header)
    my_hdu.writeto(tmpdir + "/test2.fits", overwrite=True)

    with fits.open(tmpdir + "/test1.fits") as f_orig, fits.open(tmpdir + "/test2.fits") as f_new:
        diff = f_orig[0].data - f_new[0].data[:, ::-1]
        assert np.amax(np.abs(diff)) < 1.0e-7

        # WCSs
        wcs_orig = WCS(f_orig[0].header)
        wcs_new = WCS(f_new[0].header)

        # map points
        points_orig = np.array([[100.0, 250.0], [3000.0, 800.0]])
        points_sky = wcs_orig.all_pix2world(points_orig, 0)
        points_new = wcs_new.all_world2pix(points_sky, 0)
        print(points_new)
        points_compare = np.copy(points_orig)
        points_compare[:, 0] = 4087.0 - points_compare[:, 0]
        print(np.abs(points_compare - points_new))
        err = np.amax(np.abs(points_compare - points_new))
        print(err)
        assert err < 1.0e-4
