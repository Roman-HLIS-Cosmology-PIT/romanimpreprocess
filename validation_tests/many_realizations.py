"""This is a simple test script to run a simulation many times, and report
the statistics of the output images.
"""

# Generic imports
import sys

import asdf
import numpy as np
import yaml
from astropy.io import fits

from .. import pars

# Relative from this package
from ..from_sim import sim_to_isim
from ..L1_to_L2 import gen_cal_image
from ..utils import maskhandling

# Get setup
if len(sys.argv) <= 4:
    print(
        "Calling format: python3 -m romanimpreprocess.tests.many_realizations "
        "configsim->L1 configL1->L2 N tempdir\n"
    )
    exit()
with open(sys.argv[1]) as f:
    config1 = yaml.safe_load(f)
with open(sys.argv[2]) as f:
    config2 = yaml.safe_load(f)
Nrun = int(sys.argv[3])
tempdir = sys.argv[4]

if "SEED" not in config1:
    config1["SEED"] = 100

print("--- SETUP INFROMATION ---")
print(config1)
print(config2)
print(Nrun)
print(tempdir)

if config1["OUT"] != config2["IN"]:
    print("Error: broken pipe, config1->config2")
    exit()

# get the theoretical slope (in DN/s)
slope_ideal = np.zeros((4096, 4096), dtype=np.float32)
with fits.open(config1["IN"]) as f:
    slope_ideal[4:-4, 4:-4] = f[0].data / float(f[0].header["EXPTIME"]) / pars.g_ideal  # this is in DN/s
scanum = int(config1["IN"].split("_")[-1].split(".")[0])
print("SCA", scanum)
sys.stdout.flush()
slope_ideal = slope_ideal[:, ::-1] if scanum % 3 == 0 else slope_ideal[::-1, :]

# get differences and outputs
diffs = np.memmap(tempdir + "/diffs.mmap", dtype=np.float32, mode="w+", shape=(Nrun, 4096, 4096))
images = np.memmap(tempdir + "/images.mmap", dtype=np.float32, mode="w+", shape=(Nrun, 4096, 4096))
err = np.memmap(tempdir + "/images.mmap", dtype=np.float32, mode="w+", shape=(Nrun, 4096, 4096))
moments = np.zeros((3, 4088, 4088), dtype=np.float32)

for j in range(Nrun):
    print(f"starting sim {j:d}")
    sys.stdout.flush()

    # run the simulations
    config1["SEED"] += 10
    sim_to_isim.run_config(config1)
    gen_cal_image.calibrateimage(config2)

    with asdf.open(config2["IN"]) as f:
        diffs[j, :, :] = f["roman"]["data"][-1, :, :].astype(np.float32) - f["roman"]["data"][1, :, :].astype(
            np.float32
        )

    with asdf.open(config2["OUT"]) as f:
        images[j, 4:-4, 4:-4] = f["roman"]["data"]
        err[j, 4:-4, 4:-4] = f["roman"]["err"]
        w = np.logical_not(maskhandling.PixelMask1.build(f["roman"]["dq"]))
        moments[0, :, :] += np.where(w, 1, 0.0)
        moments[1, :, :] += np.where(w, f["roman"]["data"], 0.0)
        moments[2, :, :] += np.where(w, f["roman"]["data"] ** 2, 0.0)

# number, mean & variance
moments[1:, :, :] /= moments[0, :, :] + 1e-25  # prevent div by zero
moments[2, :, :] = np.sqrt(np.clip(moments[2, :, :] - moments[1, :, :] ** 2, 0, None))
moments[1:, :, :] = np.where(moments[0, :, :][None, :, :] > 0.1, moments[1:, :, :], -1000.0)
moments_big = np.zeros((3, 4096, 4096), dtype=np.float32)
moments_big[:, 4:-4, 4:-4] = moments
del moments

# report the statistics
stack = [
    slope_ideal,
    np.median(diffs, axis=0),
    np.median(images, axis=0),
    moments_big[0, :, :],
    moments_big[1, :, :],
    moments_big[2, :, :],
    moments_big[1, :, :] - slope_ideal,
    np.median(err, axis=0),
]
out = np.memmap(tempdir + "/out.mmap", dtype=np.float32, mode="w+", shape=(len(stack), 4096, 4096))
for i in range(len(stack)):
    out[i, :, :] = stack[i]
fits.PrimaryHDU(out).writeto(config2["OUT"][:-5] + "_many_out.fits", overwrite=True)
