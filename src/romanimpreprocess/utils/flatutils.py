"""
Utilities for handling flat fields.

(Recall that if we are applying an IPC operator, the flat needs to be IPC-deconvolved.)

Functions
---------
get_flat
    Flat field in DN units, with IPC deconvolution enabled.

"""

import asdf
import numpy as np
from roman_datamodels.dqflags import pixel

from .ipc_linearity import ipc_rev


def get_flat(caldir, meta, pdq, ipc_deconvolve=True):
    """
    Gets the flat field in DN, including IPC deconvolution if requested.

    Pixels may be flagged if there are problems (ignored if pdq is None).

    Parameters
    ----------
    caldir : dict
        Locations of calibration files.
    meta : dict
        Metadata (must contain ``'nborder'``).
    pdq : np.array
        2D array, flags (pixel data quality)
    ipc_deconvolve : bool, optional
        Deconvolve the IPC?

    Returns
    -------
    np.array
        2D flat image in DN-based units.

    """

    nborder = meta["nborder"]

    # for the *active* pixels, get the flat field -- pad with 1 for the reference pixels
    with asdf.open(caldir["flat"]) as f:
        (ny, nx) = np.shape(f["roman"]["data"])
        this_flat = np.ones((ny, nx), dtype=np.float32)
        this_flat[nborder : ny - nborder, nborder : nx - nborder] = f["roman"]["data"][
            nborder : ny - nborder, nborder : nx - nborder
        ]

    # this just prevents divide-by-zero warning for the bad pixels
    if pdq is not None:
        pdq |= np.where(np.logical_or(this_flat < 0.1, this_flat > 10), pixel.NO_FLAT_FIELD, 0).astype(
            np.uint32
        )
    this_flat = np.clip(this_flat, 0.1, 10)

    if ipc_deconvolve:
        # convert to e/p/s
        with asdf.open(caldir["gain"]) as f:
            g = f["roman"]["data"][nborder : ny - nborder, nborder : nx - nborder]
            if pdq is not None:
                pdq[nborder : ny - nborder, nborder : nx - nborder] |= np.where(
                    g <= 0.1, pixel.NO_GAIN_VALUE, 0
                ).astype(np.uint32)
                g = np.clip(g, 0.1, None)
            # de-convolve the IPC kernel
            with asdf.open(caldir["ipc4d"]) as ipc:
                this_flat[nborder : ny - nborder, nborder : nx - nborder] = ipc_rev(
                    this_flat[nborder : ny - nborder, nborder : nx - nborder], ipc["roman"]["data"], gain=g
                )

    return this_flat
