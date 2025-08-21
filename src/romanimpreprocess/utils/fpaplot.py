"""Script to make a plot of properties of the focal plane from a bunch of calibration files.
"""

import os
import sys
from pathlib import Path

import asdf

# plotting
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.switch_backend("agg")

# focal plane parameters
nside_base = 4096
# the below are in units of pixels ( = 0.01 mm)
ctrs = np.array(
    [
        [2214, 1215],
        [2229, -3703],
        [2244, -8206],
        [6642, 2090],
        [6692, -2828],
        [6742, -7306],
        [11070, 4220],
        [11148, -698],
        [11264, -5106],
        [-2214, 1215],
        [-2229, -3703],
        [-2244, -8206],
        [-6642, 2090],
        [-6692, -2828],
        [-6742, -7306],
        [-11070, 4220],
        [-11148, -698],
        [-11264, -5106],
    ],
    dtype=np.int32,
)
bbox = {"xmin": -13312, "xmax": 13312, "ymin": -10254, "ymax": 6268}  # bounding box includes reference pixels


def read_sca_image(infile_format, n1, ptype, scanum, **kwargs):
    """Makes an image of the given SCA, binned to size n1,n1.

    Inputs
    -------
    infile_format : format string: input file should be infile_format.format(filestring, scanum)
    n1 : output size that we want to bin to (must be a power of 2)
    ptype : the data type
    scanum : the SCA number (1..18)

    Optional kwargs
    ----------------
    mask : builds a mask based on the indicated class

    Returns
    ---------
    arr : numpy array, binned to (n1,n1).

    Comments
    ---------
    The legal ptypes are:
    gain, alphaH, alphaV, alphaD
    """

    # find the file we need
    filestring = {
        "gain": "gain",
        "alphaH": "ipc4d",
        "alphaV": "ipc4d",
        "alphaD": "ipc4d",
        "lin2": "linearitylegendre",
        "lin3": "linearitylegendre",
        "pflatnorm": "pflat",
        "read": "read",
    }
    # and the leading coordinates (for 3D or 4D arrays)
    pos = {
        "gain": None,
        "alphaH": [1, 0],
        "alphaV": [0, 1],
        "alphaD": [0, 0],
        "lin2": [2],
        "lin3": [3],
        "pflat": None,
        "read": None,
    }

    file = infile_format.format(filestring[ptype], scanum)
    arr = np.zeros((n1, n1))
    # return now if the file we need doesn't exist
    if not os.path.exists(file):
        return arr
    # below here, we know the file exists
    with asdf.open(file) as f:
        obj = f["roman"]["data"]
        s = np.shape(obj)
        if len(s) == 2:
            use_obj = obj
        elif len(s) == 3:
            use_obj = obj[pos[ptype][0], :, :]
        elif len(s) == 4:
            use_obj = obj[pos[ptype][0], pos[ptype][1], :, :]
        else:
            raise ValueError("fpaplot.read_sca_image: Incorrect array dimension.")

        # pad if needed
        s = (nside_base - np.shape(use_obj)[0]) // 2
        if s > 0:
            use_obj = np.pad(use_obj, s)

        # mask if need be
        with asdf.open(infile_format.format("mask", scanum)) as m:
            thismask = kwargs.get("mask").build(m["roman"]["dq"])
            use_obj = np.where(np.logical_not(thismask), use_obj, np.nan)

        arr = np.nanmean(use_obj.reshape((n1, nside_base // n1, n1, nside_base // n1)), axis=(1, 3))
    return arr


lfile = str(Path(__file__).with_name("letters.dat"))
letters = np.loadtxt(lfile).reshape((256, 12, 6)).astype(np.uint16)
del lfile


def write_text(image, origin, size, val, string):
    """Utility to write text on an image."""
    (posy, posx) = origin
    for i in range(len(string)):
        if posx + size * 6 > np.shape(image)[-1] or posy + size * 12 > np.shape(image)[-2]:
            break
        card = letters[ord(string[i]), ::-1, :]
        card = np.repeat(np.repeat(card, size, axis=0), size, axis=1)
        image[posy : posy + size * 12, posx : posx + size * 6] = np.where(
            card > 0, val, image[posy : posy + size * 12, posx : posx + size * 6]
        )
        posx += size * 8


def make_big_image(infile_format, n1, ptype, **kwargs):
    """Makes an RGB image of the focal plane.

    Inputs
    -------
    infile_format : format string: input file should be infile_format.format(filestring, scanum)
    n1 : output size that we want to bin to (must be a power of 2)
    ptype : the data type

    Required kwargs
    ----------------
    vmin, vmax : scale limits

    Optional kwargs
    ----------------
    scale : display color scale

    Returns
    ---------
    arr : 3D RGB numpy array (RGB is axis=-1)
    """

    # first get the array size we need
    scale = nside_base // n1  # number of physical pixels in each apparent pixel
    nx = (bbox["xmax"] - bbox["xmin"] + 1) // scale
    ny = (bbox["ymax"] - bbox["ymin"] + 1) // scale
    arr = np.zeros((ny, nx, 3), dtype=np.uint8)
    arr[:, :, :] = 255  # make background white

    cmap = matplotlib.colormaps["viridis"]
    if "cmap" in kwargs:
        cmap = matplotlib.colormaps[kwargs.get("cmap")]

    for scanum in range(1, 19):
        myImage = read_sca_image(infile_format, n1, ptype, scanum, **kwargs)

        # pflat is special and has the per-chip median divided out
        if ptype == "pflat":
            myImage /= np.nanmedian(myImage) + 1e-24

        myImage = np.nan_to_num(myImage, 0.0)
        myImage = np.clip(
            (myImage - kwargs.get("vmin")) / (kwargs.get("vmax") - kwargs.get("vmin")), 0.0, 1.0
        )
        posx = (ctrs[scanum - 1, 0] - nside_base // 2 - bbox["xmin"]) // scale
        posy = (ctrs[scanum - 1, 1] - nside_base // 2 - bbox["ymin"]) // scale
        arr[posy : posy + n1, posx : posx + n1, :] = cmap(myImage, bytes=True)[:, :, :3]

    if "scale" in kwargs:
        arr[-(n1 // 8) :, nx // 2 - n1 : nx // 2 + n1, :] = cmap(np.linspace(0, 1, 2 * n1), bytes=True)[
            None, :, :3
        ]
        sc = max(n1, 64) // 64
        posy = ny - n1 // 8 - 15 * sc
        for j in range(3):
            arr[-(n1 // 8) - 2 * sc : -(n1 // 8), nx // 2 - n1 + j * n1 : nx // 2 - n1 + j * n1 + sc, :] = 0
            txt = kwargs.get("scale").format(
                j / 2.0 * (kwargs.get("vmax") - kwargs.get("vmin")) + kwargs.get("vmin")
            )
            posx = nx // 2 - n1 + n1 * j - 3 * sc * len(txt)
            for l_ in range(3):
                write_text(arr[:, :, l_], (posy, posx), sc, 0, txt)

        # label the scale
        label = {
            "gain": "gain (e/DN)",
            "alphaH": "IPC_h",
            "alphaV": "IPC_v",
            "alphaD": "IPC_d",
            "lin2": "c2 (DN)",
            "lin3": "c3 (DN)",
            "pflatnorm": "pflatnorm",
            "read": "rn (DN)",
        }

        posx = nx // 2 - 3 * sc * len(label[ptype])
        posy = ny - n1 // 8 - 27 * sc
        for l_ in range(3):
            write_text(arr[:, :, l_], (posy, posx), sc, 0, label[ptype])

    return arr


def multi_image(infile_format, n1, masktype):
    """Makes a multi-panel image of the focal plane."""

    my_images = []

    # linearity
    my_images.append(
        make_big_image(infile_format, n1, "lin2", vmin=-100.0, vmax=2900.0, scale="{:4.0f}", mask=masktype)
    )
    my_images.append(
        make_big_image(infile_format, n1, "lin3", vmin=-100.0, vmax=1500.0, scale="{:4.0f}", mask=masktype)
    )

    # gain
    my_images.append(
        make_big_image(infile_format, n1, "gain", vmin=1.2, vmax=2.1, scale="{:4.2f}", mask=masktype)
    )

    # IPC
    my_images.append(
        make_big_image(infile_format, n1, "alphaD", vmin=0.0, vmax=0.004, scale="{:5.3f}", mask=masktype)
    )
    my_images.append(
        make_big_image(infile_format, n1, "alphaH", vmin=0.005, vmax=0.025, scale="{:5.3f}", mask=masktype)
    )
    my_images.append(
        make_big_image(infile_format, n1, "alphaV", vmin=0.005, vmax=0.025, scale="{:5.3f}", mask=masktype)
    )

    # flat
    my_images.append(
        make_big_image(infile_format, n1, "pflatnorm", vmin=0.8, vmax=1.2, scale="{:4.2f}", mask=masktype)
    )
    # read noise
    my_images.append(
        make_big_image(infile_format, n1, "read", vmin=4.0, vmax=9.0, scale="{:4.1f}", mask=masktype)
    )

    # now the whole image
    n_image = len(my_images)
    (ny, nx, nc) = np.shape(my_images[0])
    nw = 2
    nh = (n_image - 1) // nw + 1
    gap = 1 + n1 // 4
    arr = np.zeros((ny * nh + gap * (nh - 1), nx * nw + gap * (nw - 1), nc), dtype=np.uint8)
    arr[:, :, :] = 255
    for i in range(n_image):
        posx = (i % nw) * (nx + gap)
        posy = (i // nw) * (ny + gap)
        arr[posy : posy + ny, posx : posx + nx, :] = my_images[i]

    return arr


if __name__ == "__main__":
    """Simple script to make an array"""

    try:
        from .maskhandling import PixelMask1
    except (ImportError, ModuleNotFoundError):
        from maskhandling import PixelMask1
    arr = multi_image(sys.argv[1], 128, PixelMask1)
    # arr = make_big_image('/fs/scratch/PCON0003/cond0007/cal/roman_wfi_{:s}_DUMMY20250521_SCA{:02d}.asdf',
    #      128, 'gain', vmin=1.2, vmax=2., scale='{:4.2f}', mask=PixelMask1)
    img = Image.fromarray(arr[::-1, :, :])
    img.save(sys.argv[2])
