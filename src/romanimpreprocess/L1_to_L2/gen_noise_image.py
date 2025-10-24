"""
Routines to generate noise realizations.

Functions
---------
_get_subscript
    Helper function for parsing noise directives.
make_noise_cube
    Makes a noise cube; a bunch of 2D images (realization, y, x).
generate_all_noise
    Driver to generate noise realizations.

"""

import re
import sys
from copy import deepcopy

import asdf
import galsim
import numpy as np
import yaml
from astropy.io import fits

from .. import pars
from ..from_sim.sim_to_isim import fill_in_refdata_and_1f
from ..utils import sky
from .GalPoisson.draw_with_tilnus import draw_from_Pearson
from .GalPoisson.find_tilnus import get_tilde_nus
from .gen_cal_image import calibrateimage


def _get_subscript(arr, ch):
    """
    Helper function for parsing noise directives.

    Takes a character `ch` in the array, and return the string that
    goes out to but does not include the next capital letter. e.g.::

      _get_subscript('RS2Pg4', 'S') --> '2'
      _get_subscript('RS2Pg4', 'P') --> 'g4'

    Parameters
    ----------
    arr : str
        A string containing many directives separated by capital letters.
    ch : str
        Character that we want to subscript.

    Returns
    -------
    str
        The subscript of `ch`.

    """

    return re.split(r"(?=[A-Z])", arr.split(ch)[-1])[0]


def make_noise_cube(config, rng):
    """
    Noise generator.

    Makes alternative files with extra read noise, and then differences them to return
    a "noise only" slope image. The list of noise realizations to generate is controlled
    by ``config["NOISE"]["LAYER"]``.

    Parameters
    ----------
    config : dict
        Configuration dictionary (likely unpacked from a YAML file).
    rng : galsim.BaseDeviate
        Random number generator.

    Returns
    -------
    np.array
        The noise realizations, shape = (N_noise,nside_active,nside_active).

    """

    # The number of noise realizations we need
    N_noise = len(config["NOISE"]["LAYER"])
    noiseimage = np.zeros((N_noise, pars.nside_active, pars.nside_active), dtype=np.float32)

    # For each realization, we will load the L1 data file, add read noise, and write to file.
    for i_noise in range(N_noise):
        cmd = config["NOISE"]["LAYER"][i_noise]  # this noise layer command
        with asdf.open(config["IN"]) as f_in:
            mytree = deepcopy(f_in.tree)  # load 'old' data from disk
        # orig = np.copy(mytree["roman"]["data"])  # and make a copy that we won't modify <-- not needed
        nb = pars.nborder  # shorthand for border width

        with asdf.open(config["OUT"]) as f_orig:
            diff = np.zeros_like(f_orig["roman"]["data"])

        # read noise simulated?
        if "R" in cmd:
            noiseflags = _get_subscript(cmd, "R")  # get information on what to simulate

            # if not adding, clear the input image
            if "a" not in noiseflags:
                with asdf.open(config["CALDIR"]["dark"]) as fb:
                    mytree["roman"]["data"] = np.copy(fb["roman"]["data"]).astype(
                        mytree["roman"]["data"].dtype
                    )

                # write this to a file and calibrate it
                with asdf.AsdfFile(mytree) as af, open(config["NOISE"]["TEMP"], "wb") as f:
                    af.write_to(f)
                config3 = deepcopy(config)
                config3["IN"] = config["NOISE"]["TEMP"]
                config3["OUT"] = config["NOISE"]["TEMP"][:-5] + "_refL2.asdf"
                calibrateimage(config3)

            # white noise
            for k in range(len(mytree["roman"]["meta"]["exposure"]["read_pattern"])):
                resultants = np.copy(mytree["roman"]["data"][k, nb:-nb, nb:-nb].astype(np.float32))
                im = np.zeros_like(resultants)
                galsim.GaussianDeviate(rng).generate(im)
                with asdf.open(config["CALDIR"]["read"]) as fr:
                    im *= fr["roman"]["data"][nb:-nb, nb:-nb] / np.sqrt(
                        len(mytree["roman"]["meta"]["exposure"]["read_pattern"][k])
                    )
                resultants += im
                del im
                mytree["roman"]["data"][k, nb:-nb, nb:-nb] = np.round(
                    np.clip(resultants, 0, 2**16 - 1)
                ).astype(mytree["roman"]["data"].dtype)
                del resultants

            # correlated noise
            fill_in_refdata_and_1f(
                mytree["roman"]["data"],  # the data array
                config["CALDIR"],  # calibration data structure
                rng,  # random number generator
                mytree["roman"]["meta"]["exposure"]["read_pattern"],  # readout scheme
                amp33=mytree["roman"]["amp33"],  # reference output
            )

            # write to a temporary file
            with asdf.AsdfFile(mytree) as af, open(config["NOISE"]["TEMP"], "wb") as f:
                af.write_to(f)

            # now run L1-->L2 for this file
            config2 = deepcopy(config)
            config2["IN"] = config["NOISE"]["TEMP"]
            config2["OUT"] = config["NOISE"]["TEMP"][:-5] + "_L2.asdf"
            calibrateimage(config2)

            # get difference
            with asdf.open(config2["OUT"]) as f_out:
                origfile = config["OUT"]
                if "a" not in noiseflags:
                    origfile = config3["OUT"]
                with asdf.open(origfile) as f_orig:
                    diff = f_out["roman"]["data"] - f_orig["roman"]["data"]

            # clip if requested
            if "z" in noiseflags:
                zclip = float(_get_subscript(noiseflags.upper(), "Z"))
                IQR = np.percentile(diff, 75) - np.percentile(diff, 25)
                MED = np.percentile(diff, 50)
                print("***", noiseflags, zclip, IQR, MED)
                diff = np.clip(diff, MED - zclip * IQR / 1.34896, MED + zclip * IQR / 1.34896)

        # Noise realizations for pseudo-Poisson noise bias corrections
        if "O" in cmd:
            # Get gain
            with asdf.open(config["CALDIR"]["gain"]) as g_:
                gain = np.clip(g_["roman"]["data"], 1e-4, 1e4)  # prevent division by zero error
            with asdf.open(config["OUT"]) as f_orig:
                # trim if needed (removes reference pixels --- if gain is the full array, will have d=4)
                d = (np.shape(gain)[-1] - np.shape(f_orig["roman"]["data_withsky"])[-1]) // 2
                if d > 0:
                    gain = gain[d:-d, d:-d]
                gI = gain * f_orig["roman"]["data_withsky"].value

            # ramp-fitting weights
            ngrp = len(mytree["roman"]["meta"]["exposure"]["read_pattern"])
            weightvecs = [""] * ngrp
            with asdf.open(config["OUT"]) as f_L2:
                meta = f_L2["processinfo"]["meta"]
                weightvecs[-1] = np.copy(f_L2["processinfo"]["weights"])
                start = 0
                if f_L2["processinfo"]["exclude_first"]:
                    start = 1
                for iend in range(start + 2, ngrp):
                    Kt = np.zeros(ngrp, dtype=np.float32)
                    Kt[iend - 1] = 1.0 / (meta["tbar"][iend - 1] - meta["tbar"][start])
                    Kt[start] = -Kt[iend - 1]
                    weightvecs[iend - 1] = Kt
                endslice = np.where(
                    f_L2["processinfo"]["endslice"] > 0, f_L2["processinfo"]["endslice"], ngrp - 1
                )
                noise_array = np.zeros_like(endslice, dtype=np.float32)

                a_beta = np.empty(ngrp, dtype=int)
                N_beta = np.empty(ngrp, dtype=int)
                for i in range(ngrp):
                    a_beta[i] = f_L2["processinfo"]["meta"]["read_pattern"][i][0]
                    N_beta[i] = len(f_L2["processinfo"]["meta"]["read_pattern"][i])

            for i in range(start+1, ngrp):
                tilnu21, tilnu31, tilnu41, tilnu42 = get_tilde_nus(N_beta, a_beta, weightvecs[i])

                pixels = np.where(endslice == i)

                noise_array[pixels] = draw_from_Pearson(
                    tilnu21, tilnu31, tilnu41, gI[pixels], rng=rng.as_numpy_generator()
                )

            diff[:, :] += noise_array / gain

        # Poisson noise simulated?
        if "P" in cmd:
            noiseflags = _get_subscript(cmd, "P")  # get information on what to simulate

            # first get the sky map. The 'b' flag chooses background only (with numerical order, if given).
            if "b" in noiseflags:
                sky_order = int("0" + _get_subscript(noiseflags.upper(), "B"))
                with asdf.open(config["OUT"]) as f_orig:
                    skylevel = sky.medfit(f_orig["roman"]["data_withsky"].value, order=sky_order)[1]
            else:
                with asdf.open(config["OUT"]) as f_orig:
                    skylevel = np.copy(f_orig["roman"]["data_withsky"].value)

            # ramp-fitting weights
            ngrp = len(mytree["roman"]["meta"]["exposure"]["read_pattern"])
            weightvecs = [""] * ngrp
            with asdf.open(config["OUT"]) as f_L2:
                meta = f_L2["processinfo"]["meta"]
                weightvecs[-1] = np.copy(f_L2["processinfo"]["weights"])
                start = 0
                if f_L2["processinfo"]["exclude_first"]:
                    start = 1
                for iend in range(start + 2, ngrp):
                    Kt = np.zeros(ngrp, dtype=np.float32)
                    Kt[iend - 1] = 1.0 / (meta["tbar"][iend - 1] - meta["tbar"][start])
                    Kt[start] = -Kt[iend - 1]
                    weightvecs[iend - 1] = Kt
                endslice_ = np.where(
                    f_L2["processinfo"]["endslice"] > 0, f_L2["processinfo"]["endslice"], ngrp - 1
                )
                endslice = endslice_  # noqa: F841
            # At this point, weightvecs is a list of 1D numpy arrays.
            # So the weight that went into sample j_samp in pixel (x,y) is
            # weightvecs[endslice[y,x]][j_samp]

            print("weightvecs =", weightvecs)
            print("endslice =", endslice, np.shape(endslice))
            sys.stdout.flush()

            if "r" in noiseflags:
                # generates re-sampled Poisson with the right variance

                # get the gain map
                with asdf.open(config["CALDIR"]["gain"]) as g_:
                    gain = np.clip(g_["roman"]["data"], 1e-4, 1e4)  # prevent division by zero error
                # trim if needed
                d = (np.shape(gain)[-1] - np.shape(skylevel)[-1]) // 2
                if d > 0:
                    gain = gain[d:-d, d:-d]
                n = np.shape(skylevel)[-1]

                lastsamp = mytree["roman"]["meta"]["exposure"]["read_pattern"][-1][-1]
                e_per_slice = skylevel * gain * mytree["roman"]["meta"]["exposure"]["frame_time"]
                delta_resultants = np.zeros((ngrp, n, n), dtype=np.float32)

                print("e_per_slice[2:6,2:6] =", e_per_slice[2:6, 2:6])
                for j in [1, 5, 25, 50, 75, 95, 99]:
                    print(f"{j:2d} %ile {np.percentile(e_per_slice, j)}")
                e_per_slice = np.clip(e_per_slice, 0.0, None)  # eliminate issue with zeros

                current_sample = np.zeros(np.shape(e_per_slice), dtype=np.float32)

                for isamp in range(lastsamp + 1):
                    # get Poisson error in that slice
                    sample = np.copy(e_per_slice.astype(np.float64))
                    galsim.PoissonDeviate(rng).generate_from_expectation(sample)
                    sample -= e_per_slice
                    sample /= gain  # convert to DN
                    current_sample += sample

                    print(">>", isamp, current_sample[2:6, 2:6])
                    sys.stdout.flush()

                    # build the table of changes in resultant
                    for j in range(ngrp):
                        if isamp in mytree["roman"]["meta"]["exposure"]["read_pattern"][j]:
                            delta_resultants[j, :, :] += current_sample / len(
                                mytree["roman"]["meta"]["exposure"]["read_pattern"][j]
                            )

                # now these resultants are in DN

                print(delta_resultants[:, 2:6, 2:6])
                sys.stdout.flush()

                # ramp fit
                for es in range(ngrp):
                    if isinstance(weightvecs[es], np.ndarray):
                        print("es =", es, weightvecs[es])
                        sys.stdout.flush()
                        for j in range(len(weightvecs[es])):
                            diff[:, :] += np.where(
                                endslice == es, weightvecs[es][j] * delta_resultants[j, :, :], 0.0
                            )

        # remove modes that would be taken out in sky subtraction
        if "S" in cmd:
            sky_order = int("0" + _get_subscript(cmd, "S"))
            diff -= sky.medfit(diff, order=sky_order)[1]

        noiseimage[i_noise, :, :] = diff

    return noiseimage


def generate_all_noise(config):
    r"""
    Driver for noise generation.

    Parameters
    ----------
    config : dict
        The configuration (usually unpacked from a YAML file).

    Returns
    -------
    None

    Notes
    -----
    This requires an additional 'NOISE' object in the configuration dictionary.
    It should have the entries:

    * ``config['NOISE']['LAYER']`` : list of noise realizations to build
    * ``config['NOISE']['TEMP']`` : temporary noise file location
    * ``config['NOISE']['SEED']`` : random number seed for the read noise images
    * ``config['NOISE']['OUT']`` : output of noise cube

    Layer commands start with a capital letter, then have lower case or numerical indications:

    * ``R`` = include read noise
    * ``P...`` = placeholder for Poisson noise
    * ``S\d*`` = subtract sky using median filter of given order
    * ``C...`` = reserved for comment (no capital letters in ...)

    The configuration has to have been run, since it looks for the L2 file written
    by ``gen_cal_image.calibrateimage``.

    """

    rng = galsim.UniformDeviate(config["NOISE"]["SEED"])
    noiseimage = make_noise_cube(config, rng)

    print(np.shape(noiseimage))
    print("percentiles:")
    for q in [5, 25, 50, 75, 95]:
        print(q, np.percentile(noiseimage, q, axis=(1, 2)))

    if "NOISE_PRECISION" in config:
        if config["NOISE_PRECISION"] == 16:
            noiseimage = noiseimage.astype(np.float16)
        if config["NOISE_PRECISION"] not in [16, 32]:
            raise ValueError("Unsupported noise precision.")

    # now output the noise image
    tree = {"config": config, "noise": noiseimage}
    with asdf.AsdfFile(tree) as af, open(config["NOISE"]["OUT"], "wb") as f:
        af.write_to(f)
    if "FITSOUT" in config:
        if config["FITSOUT"]:
            fitsout = fits.HDUList([fits.PrimaryHDU(noiseimage)])
            fitsout.writeto(config["NOISE"]["OUT"][:-5] + "_asdf_to.fits", overwrite=True)


if __name__ == "__main__":
    """Stand-alone function processes L1->L2 and generates noise"""

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    calibrateimage(config | {"SLICEOUT": True})  # add slice information
    generate_all_noise(config)
