"""Makes gain and associated reference files."""

import sys
from datetime import UTC, datetime
from os.path import split as pathsplit

import asdf
import numpy as np
from astropy.io import fits
from scipy.signal import convolve

### Command-line inputs to this script ###

summaries = sys.argv[1]  # summary files
sca = int(sys.argv[2])  # the SCA number
outfile = sys.argv[3]  # output file

### End command-line inputs ###

# solid-waffle output columns
cols = {"X": 0, "Y": 1, "N": 2, "g": 5, "aH": 6, "aV": 7, "aD": 10}

nside = 4096  # dimension of H4RG

with open(summaries) as FF:
    infiles = FF.readlines()
infiles = [f.rstrip() for f in infiles]
N_in = len(infiles)

# now load the files
for j in range(N_in):
    print(infiles[j])
    data = np.loadtxt(infiles[j])
    (nrow, ncol) = np.shape(data)
    if j == 0:
        alldata = np.zeros((N_in, nrow, ncol))
    alldata[j, :, :] = data

good = np.count_nonzero(alldata[:, :, cols["N"]], axis=0) > 0
nx = 1 + int(np.amax(alldata[0, :, cols["X"]]))
ny = 1 + int(np.amax(alldata[0, :, cols["Y"]]))
print("superpixels", nx, ny)
rx = nside // nx
ry = nside // ny
print("repeat", rx, ry)
print(good, "-->", np.count_nonzero(good), "good pixels")

meanvals = {}
tmean = {}
# get the gain and IPC data, with array means filled in for bad pixels
for e in ["g", "aH", "aV", "aD"]:
    meanvals[e] = np.nanmean(np.where(alldata[:, :, cols["N"]] > 0, alldata[:, :, cols[e]], np.nan), axis=0)
    tmean[e] = np.nanmean(meanvals[e])
    meanvals[e] = np.where(good, meanvals[e], tmean[e])

print("mean values", tmean)


def unpack(X):
    """Helper function to make nside x nside arrays."""

    temp = np.repeat(np.repeat(X.reshape((ny, nx)), ry, axis=0), rx, axis=1)
    # set reference pixels to zero
    temp[:, :4] = 0.0
    temp[:, -4:] = 0.0
    temp[:4, :] = 0.0
    temp[-4:, :] = 0.0
    return temp


good_unpack = unpack(good)

# get the configuration file inputs
config_lines = []
for j in range(N_in):
    iq = infiles[j][:-11] + "config.txt"
    config_lines.append("# " + iq)
    with open(iq) as FF:
        sl = FF.readlines()
    config_lines.extend([s.rstrip() for s in sl])
config_lines = "\n".join(config_lines)  # turn into a string
print("--")
print(config_lines)
print("--")

### write gain file ###

gain_unpack = unpack(meanvals["g"]).astype(np.float32)

tree = {
    "roman": {
        "meta": {
            "author": "make_gain_file.py",
            "description": "make_gain_file.py",
            "instrument": {"detector": f"WFI{sca:02d}", "name": "WFI"},
            "origin": "PIT - romanimpreprocess",
            "date": datetime.now(UTC).isoformat(),
            "pedigree": "DUMMY",
            "reftype": "GAIN",
            "telescope": "ROMAN",
            "useafter": "!time/time-1.2.0 2020-01-01T00:00:00.000",
        },
        "data": gain_unpack,
        "dq": np.where(good_unpack, 0, 2**19).astype(
            np.uint32
        ),  # simple quality flag for bookkeeping for now
    },
    "notes": {"solid_waffle_config": config_lines},
}

# write to a file
asdf.AsdfFile(tree).write_to(outfile)
# this stuff is just so that we can also display the images in ds9
fits.HDUList(
    [
        fits.PrimaryHDU(),
        fits.ImageHDU(tree["roman"]["data"]),
    ]
).writeto(outfile[:-5] + "_asdf_data.fits", overwrite=True)

### write IPC file ###

# this one doesn't have a standard form yet
# in the pipeline, so I'm recycling the one from the DC2
# simulation for now.

# size is (3,3,4088,4088)
# [1+dy,1+dx,y,x] is cross talk *from* (x,y) *to* (x+dx,y+dy)

Kernel_good = (convolve(np.where(good_unpack, 0, 1), np.zeros((3, 3)), mode="same", method="direct") < 0.5)[
    4:-4, 4:-4
]

alphaH_unpack = unpack(meanvals["aH"]).astype(np.float32)[4:-4, 4:-4]
alphaV_unpack = unpack(meanvals["aV"]).astype(np.float32)[4:-4, 4:-4]
alphaD_unpack = unpack(meanvals["aD"]).astype(np.float32)[4:-4, 4:-4]

Kernel = np.zeros((3, 3, nside - 8, nside - 8))
Kernel[1, 0, :, :] = Kernel[1, 2, :, :] = alphaH_unpack
Kernel[0, 1, :, :] = Kernel[2, 1, :, :] = alphaV_unpack
Kernel[0, 0, :, :] = Kernel[2, 2, :, :] = alphaD_unpack
Kernel[0, 2, :, :] = Kernel[2, 0, :, :] = alphaD_unpack

# clip values that go off the edges since we don't find
# capacitive coupling to the reference pixels
for dy in range(-1, 2):
    for dx in range(-1, 2):
        if dy < 0:
            Kernel[1 + dy, 1 + dx, :-dy, :] = 0.0
        if dy > 0:
            Kernel[1 + dy, 1 + dx, -dy:, :] = 0.0
        if dx < 0:
            Kernel[1 + dy, 1 + dx, :, :-dx] = 0.0
        if dx > 0:
            Kernel[1 + dy, 1 + dx, :, -dx:] = 0.0

# the correlation-based one should be symmetrized
dy_ = [1, 0, 1, 1]
dx_ = [0, 1, 1, -1]
for j in range(4):
    dy = dy_[j]
    dx = dx_[j]
    ymin = max(0, -dy)
    ymax = nside - 8 + ymin - abs(dy)
    xmin = max(0, -dx)
    xmax = nside - 8 + xmin - abs(dx)
    KernelSym = (
        Kernel[1 + dy, 1 + dx, ymin:ymax, xmin:xmax]
        + Kernel[1 - dy, 1 - dx, ymin + dy : ymax + dy, xmin + dx : xmax + dx]
    ) / 2.0
    Kernel[1 + dy, 1 + dx, ymin:ymax, xmin:xmax] = KernelSym
    Kernel[1 - dy, 1 - dx, ymin + dy : ymax + dy, xmin + dx : xmax + dx] = KernelSym
# set the central value to whatever is left
Kernel[1, 1, :, :] = 0.0
Kernel[1, 1, :, :] = 1.0 - np.sum(Kernel, axis=(0, 1))

# write to file
head, tail = pathsplit(outfile)
outfile2 = head + "/" + tail.replace("_gain_", "_ipc4d_")

tree = {
    "roman": {
        "meta": {
            "author": "make_gain_file.py",
            "description": "make_gain_file.py",
            "instrument": {"detector": f"WFI{sca:02d}", "name": "WFI"},
            "origin": "PIT - romanimpreprocess",
            "date": datetime.now(UTC).isoformat(),
            "pedigree": "DUMMY",
            "reftype": "IPC4D",
            "telescope": "ROMAN",
            "useafter": "!time/time-1.2.0 2020-01-01T00:00:00.000",
        },
        "data": Kernel,
        "dq": np.where(Kernel_good, 0, 1).astype(np.uint32),  # simple quality flag for bookkeeping for now
    },
    "notes": {"solid_waffle_config": config_lines},
}

asdf.AsdfFile(tree).write_to(outfile2)
# this stuff is just so that we can also display the images in ds9
fits.HDUList(
    [
        fits.PrimaryHDU(),
        fits.ImageHDU(
            np.transpose(tree["roman"]["data"], axes=(0, 2, 1, 3)).reshape((3 * (nside - 8), 3 * (nside - 8)))
        ),
    ]
).writeto(outfile2[:-5] + "_asdf_data.fits", overwrite=True)
