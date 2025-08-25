"""
IPC and linearity utilities.

Classes
-------
IL
    IPC+Inverse linearity class.

Functions
---------
ipc_fwd
    Carries out an IPC operation on the image.
ipc_rev
    Inverse IPC operation, to the given order.
correct_cube
    IPC corrects a full data cube (data) in place.
_lin
    Helper function to evaluate Legendre-based function.
linearity
    Performs a linearity correction.
multilin
    Performs a linearity correction, but with multiple groups.
invlinearity
    Calculates the inverse linearity. (This is most likely to be used in simulations.)
test__lin
    Simple test function for _lin.
test_lin_ilin
    Forward-backward test for linearity routines.
test_IL
    Some tests for the inverse linearity class.

"""

import sys

import asdf
import numpy as np
from roman_datamodels.dqflags import pixel

## IPC utilities ##


def ipc_fwd(image, kernel, gain=None):
    """
    Carries out an IPC operation on the image.

    Parameters
    ----------
    image : np.array
        2D numpy array of size (ny,nx).
    kernel : np.array
        4D numpy array of size (3,3,ny,nx).
    gain : np.array or None, optional
        If not None, the input gain map (in e/DN).

    Returns
    -------
    output : np.array
        2D numpy array, same shape as `image`, IPC-convolved.

    See Also
    --------
    ipc_rev : Inverse function.

    Notes
    -----
    The output image, in psuedocode, is::

        output[y,x] = sum_{dy,dx} input[y-dy,x-dx] kernel[1+dy,1+dx,y-dy,x-dx]

    This function natively works in electrons, but if gain is provided then works in DN (does g^-1 K g).

    """

    im = image
    if gain is not None:
        im = gain * image

    # start with the center image
    output = im * kernel[1, 1, :, :]

    # nearest neighbors
    # dy=1, dx=0
    output[1:, :] += im[:-1, :] * kernel[2, 1, :-1, :]
    # dy=-1, dx=0
    output[:-1, :] += im[1:, :] * kernel[0, 1, 1:, :]
    # dy=0, dx=1
    output[:, 1:] += im[:, :-1] * kernel[1, 2, :, :-1]
    # dy=0, dx=-1
    output[:, :-1] += im[:, 1:] * kernel[1, 0, :, 1:]

    # diagonals
    # dy=1, dx=1
    output[1:, 1:] += im[:-1, :-1] * kernel[2, 2, :-1, :-1]
    # dy=1, dx=-1
    output[1:, :-1] += im[:-1, 1:] * kernel[2, 0, :-1, 1:]
    # dy=-1, dx=1
    output[:-1, 1:] += im[1:, :-1] * kernel[0, 2, 1:, :-1]
    # dy=-1, dx=-1
    output[:-1, :-1] += im[1:, 1:] * kernel[0, 0, 1:, 1:]

    if gain is not None:
        output /= gain

    return output


def ipc_rev(image, kernel, order=2, gain=None):
    """
    Inverse IPC operation, to the given order.

    Grows the footprint of each pixel to ``(2*order+1,2*order+1)``.

    If `gain `is provided, then does g^-1 K^-1 g image instead of K^-1 image.
    (Equivalent: operate on DN instead of e.)

    Parameters
    ----------
    image : np.array
        2D numpy array of size (ny,nx).
    kernel : np.array
        4D numpy array of size (3,3,ny,nx).
    order : int, optional
        The order of IPC inversion; default is 2nd order (correct to next-to-next-to-neighbor
        pixel). Error is of order ``alpha**(order+1)``.
    gain : np.array or None, optional
        If not None, the input gain map (in e/DN).

    Returns
    -------
    output : np.array
        2D numpy array, same shape as `image`, IPC-convolved.

    See Also
    --------
    ipc_fwd : Forward function.

    """

    image2 = image
    if gain is not None:
        image2 = gain * image
    output = np.copy(image2)
    for _ in range(order):
        output = output + image2 - ipc_fwd(output, kernel)
    if gain is not None:
        output /= gain
    return output


def correct_cube(data, ipc_file, mylog, gain_file=None):
    """
    IPC corrects a full data cube (data) in place.

    Operates in electrons if `gain_file` is None or missing, but
    operates in DN if `gain_file` is provided.

    Parameters
    ----------
    data : np.array
        3D data cube, shape (ngrp, ny, nx).
    ipc_file : str
        Location of ASDF ipc4d calibration reference file.
    mylog : romanimpreprocess.utils.processlog.ProcessLog
        Processing log.
    gain_file : str or None, optional
        If not none, the ASDF gain calibration reference file.

    """

    if ipc_file is None:
        if mylog is not None:
            mylog.append("No IPC file specified, skipping ...\n")
        return

    with asdf.open(ipc_file) as F:
        kernel = F["roman"]["data"]
        if mylog is not None:
            mylog.append(
                f"IPC kernel center range --> {np.amin(kernel[1,1,:,:]):f},{np.amax(kernel[1,1,:,:]):f}\n"
            )
        (ngrp, ny, nx) = np.shape(data)
        nb = (8192 + (nx - np.shape(kernel)[-1]) // 2) % 16
        if mylog is not None:
            mylog.append(f" ..., {ngrp:d} groups, excluding {nb:d} border pixels\n")
        if gain_file is None:
            g = 1.0
        else:
            with asdf.open(gain_file) as G:
                g = np.copy(G["roman"]["data"][nb : ny - nb, nb : nx - nb])
        for i in range(ngrp):
            data[i, nb : ny - nb, nb : nx - nb] = ipc_rev(data[i, nb : ny - nb, nb : nx - nb] * g, kernel) / g


## LINEARITY UTILITIES ##


def _lin(z, coefs, linextrap=True):
    """
    Helper function to evaluate Legendre-based function.

    Parameters
    ----------
    z : np.array
        Rescaled signal (modified DN), shape (ny,nx).
    coefs : np.array
        Legendre polynomial coefficients, shape (p_order+1,ny,nx).
    linextrap : bool, optional
        Linearly extrapolate beyond end of range?
        (Default = True is better behaved.)

    Returns
    -------
    phi : np.array
        The linearized signal, `` sum_l coefs_l P_p(z)``, shape (ny,nx)
    exflag : np.array of bool
        Extrapolated beyond |z|=1?, shape (ny,nx)

    """

    exflag = np.abs(z) > 1  # are we extrapolating?
    phi = np.copy(coefs[0, :, :])
    poly_prev = np.ones_like(phi)
    poly = np.copy(z)
    for L in range(1, np.shape(coefs)[0]):
        if linextrap:
            phi += coefs[L, :, :] * np.where(
                exflag, np.sign(z) ** L * (1 + L * (L + 1) / 2.0 * (np.abs(z) - 1)), poly
            )
        else:
            phi += coefs[L, :, :] * poly
        # Legendre polynomial recursion relation
        poly_next = (2 * L + 1) / (L + 1) * z * poly - L / (L + 1) * poly_prev
        poly_prev = poly
        poly = poly_next

    return phi, exflag


def linearity(S, linearity_file, origin=(0, 0)):
    """
    Performs a linearity correction.

    Parameters
    ----------
    S : np.array
        2D data array in DN_raw.
    linearity_file : str
        ASDF file with linearity data.
    origin : (int, int), optional
        The (x,y) position of the lower-left corner of S in the convention of the file.

    Returns
    -------
    Slin : np.array
        2D data array in DN_lin. Same shape as `S`.
    dq : np.array of uint32
        2D flag array. Same shape as `S`.

    Notes
    -----
    The coordinate convention is that if you have a block `S` that corresponds to region
    ``[128:132,256:260]``, then you would give ``origin=(256,128)``.

    """

    (dy, dx) = np.shape(S)
    ymin = origin[1]
    ymax = ymin + dy
    xmin = origin[0]
    xmax = xmin + dx

    with asdf.open(linearity_file) as F:
        Smin = F["roman"]["Smin"][ymin:ymax, xmin:xmax]
        Smax = F["roman"]["Smax"][ymin:ymax, xmin:xmax]
        phi, exflag = _lin(-1 + 2 * (S - Smin) / (Smax - Smin), F["roman"]["data"][:, ymin:ymax, xmin:xmax])
        dq = np.copy(F["roman"]["dq"][ymin:ymax, xmin:xmax])
    dq |= np.where(exflag, pixel.NO_LIN_CORR, 0).astype(np.uint32)  # flag with bad linearity correction
    return phi, dq


def multilin(S, linearity_file, origin=(0, 0), do_not_flag_first=True, attempt_corr=None):
    """
    Performs a linearity correction, but with multiple groups.

    Parameters
    ----------
    S : np.array
        Input data as a 3D numpy array, shape (ngrp, ny, nx).
    linearity_file : str
        ASDF calibration reference file with linearity data.
    origin : (int, int), optional
        The (x,y) position of the lower-left corner of `S` in the convention of the file.
    do_not_flag_first : bool, optional
        Don't flag the first read if it is out of range (True by default for reset-read
        frames that we won't use anyway).
    attempt_corr : np.array of bool, optional
        If provided, an array of the same shape as `S` that is True if we want to try the correction
        and False otherwise (the idea is that we want to be able to *not* flag a pixel that is saturated).
        Default is to attempt to correct everything.

    Returns
    -------
    Slin : np.array
        Linearized data, shape (ngrp,ny,nx), in DN_lin.
    dq : np.array of uint32
        Flag array (2D).

    Notes
    -----
    The coordinate convention is that if you have a block `S` that corresponds to region
    ``[128:132,256:260]``, then you would give ``origin=(256,128)``. This is in 2D, so
    `origin` has only 2 entries (no offset is given on the time axis).

    ``Slin=0`` corresponding to the bias level ``Sref`` in the calibration reference file.

    """

    (ngrp, dy, dx) = np.shape(S)
    ymin = origin[1]
    ymax = ymin + dy
    xmin = origin[0]
    xmax = xmin + dx

    # accept everything if attempt_corr not provided
    if attempt_corr is None:
        attempt_corr = np.ones((ngrp, dy, dx), dtype=bool)

    phi = np.zeros(np.shape(S), dtype=np.float32)
    with asdf.open(linearity_file) as F:
        Smin = F["roman"]["Smin"][ymin:ymax, xmin:xmax]
        Smax = F["roman"]["Smax"][ymin:ymax, xmin:xmax]
        Sref = F["roman"]["Sref"][ymin:ymax, xmin:xmax]
        dq = np.copy(F["roman"]["dq"][ymin:ymax, xmin:xmax])
        for j in range(ngrp):
            z = -1 + 2 * (S[j, :, :] - Smin) / (Smax - Smin)
            if j == 0 and do_not_flag_first:
                z = np.clip(z, -1, 1)
            phi[j, :, :], exflag = _lin(z, F["roman"]["data"][:, ymin:ymax, xmin:xmax])
            phi[j, :, :] = np.where(
                dq & (pixel.NO_LIN_CORR | pixel.REFERENCE_PIXEL) == 0, phi[j, :, :], S[j, :, :] - Sref
            )

            # flag reads with bad linearity correction
            if not (j == 0 and do_not_flag_first):
                dq |= np.where(np.logical_and(exflag, attempt_corr[j, :, :]), pixel.NO_LIN_CORR, 0).astype(
                    np.uint32
                )

    return phi, dq


def invlinearity(Slin, linearity_file, origin=(0, 0)):
    """
    Calculates the inverse linearity. (This is most likely to be used in simulations.)

    Parameters
    ----------
    Slin : np.array
        2D input data, shape (ny,nx), in DN_lin.
    linearity_file : str
        ASDF calibration reference file with linearity data.
    origin : (int, int), optional
        The (x,y) position of the lower-left corner of `S` in the convention of the calibration file.

    Returns
    -------
    S : np.array
        2D array, same shape as `Slin`, in DN_raw.
    exflag : np.array of bool
        Extrapolation flag.

    Notes
    -----
    This function works by bisection. It is the slowest step in the simulation -> Level 1 workflow,
    so we plan to implement a more advanced algorithm in the future.

    """

    (dy, dx) = np.shape(Slin)
    ymin = origin[1]
    ymax = ymin + dy
    xmin = origin[0]
    xmax = xmin + dx

    with asdf.open(linearity_file) as F:
        z = np.zeros_like(Slin)
        # binary search, robust over the range -1 < z < +1
        # (which should encapsulate anything; also automatically saturates)
        for j in range(1, 25):
            phi, exflag = _lin(z, F["roman"]["data"][:, ymin:ymax, xmin:xmax], linextrap=False)
            # linextrap=False saves some time
            z += np.where(phi < Slin, 1 / 2**j, -1 / 2**j)
        Smin = F["roman"]["Smin"][ymin:ymax, xmin:xmax]
        Smax = F["roman"]["Smax"][ymin:ymax, xmin:xmax]
        S = Smin + (Smax - Smin) / 2.0 * (1 + z)

    return S, exflag


"""IPC + inverse linearity forward modeling tools"""


class IL:
    """
    IPC+Inverse linearity class.

    This exists to wrap the IPC and inverse-linearity operations in a way that is consistent with
    ``romanisim``.

    Parameters
    ----------
    linearity_file : str
        The ASDF linearity calibration reference file.
    gain_file : str
        The ASDF gain calibration reference file.
    ipc_file : str or None
        The ASDF ipc4d calibration reference file.
        If None, skips the IPC.
    start_e : np.array or float, optional
        If provided, starts with some number of electrons (start_e, number or array) in the well.
        Useful for reset noise.

    Methods
    -------
    __init__
        Constructor
    set_dq
        Sets the 3D data quality flags.
    apply
        Converts a linearized signal to a non-linear, IPC-convolved signal.

    """

    def __init__(self, linearity_file, gain_file, ipc_file, start_e=0.0):
        self.linearity_file = linearity_file
        self.gain_file = gain_file
        self.ipc_file = ipc_file
        self.start_e = start_e
        # need the .dq attribute
        with asdf.open(self.linearity_file) as f:
            self._dq = np.copy(f["roman"]["dq"])

    def set_dq(self, ngroup=1, nborder=4):
        """
        Sets the 3D data quality flags.

        This is so that the data quality flags can be propagated.

        Parameters
        ----------
        ngroup : int, optional
            Number of groups to initialize.
        nborder : int, optional
            Number of border reference pixels.

        Returns
        -------
        None

        """

        (ny, nx) = np.shape(self._dq)
        self.dq = np.zeros((ngroup, ny - 2 * nborder, nx - 2 * nborder), dtype=np.uint32)
        self.dq[:, :, :] = self._dq[None, nborder : ny - nborder, nborder : nx - nborder]

    def apply(self, counts, electrons=False, electrons_out=False):
        """
        Converts a linearized signal to a non-linear signal.

        Parameters
        ----------
        counts : np.array
            2D array of counts.
        electrons : bool, optional
            Is input in electrons (True) or DN_lin (False, default).
        electrons_out : bool, optional
            Is output in electrons (True) or DN_raw (False, default).

        """

        print("apply", electrons, electrons_out, np.shape(counts))
        # print(counts[:6, :6])
        sys.stdout.flush()

        # this uses a 4d IPC file
        if self.ipc_file is not None:
            with asdf.open(self.ipc_file) as f:
                counts_conv = ipc_fwd(counts + self.start_e, f["roman"]["data"])
        else:
            counts_conv = counts + self.start_e

        # gain factors for in and out
        (nyc, nxc) = np.shape(counts)
        g_in = 1.0
        g_out = 1.0
        if electrons or electrons_out:
            with asdf.open(self.gain_file) as f:
                # extract the gain (with reference pixels clipped if needed)
                g = f["roman"]["data"]
                (nyg, nxg) = np.shape(g)
                if nyg > nyc:
                    nb = (nyg - nyc) // 2
                    g = g[nb:-nb, nb:-nb]
            if electrons:
                g_in = g
            if electrons_out:
                g_out = g

        # what to strip off the counts array
        nb = (8192 - nyc // 2) % 16
        S, _ = invlinearity(counts_conv / g_in, self.linearity_file, origin=(nb, nb))

        if not electrons_out:
            return S

        # below here, we know electrons_out is on.
        with asdf.open(self.linearity_file) as F:
            return g_out * (S - F["roman"]["Sref"][nb : nb + nyc, nb : nb + nxc])


def test__lin():
    """Simple test function for _lin."""
    z = np.linspace(-1.5, 1.5, 31).reshape((1, 31))
    coefs = np.zeros((4, 1, 31))
    coefs[3, :, :] = 1.0
    phi, _ = _lin(z, coefs)
    print(phi)


def test_lin_ilin(linearity_file):
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
        S = F["roman"]["Sref"][ymin:ymax, xmin:xmax] + np.linspace(0, dx * dy - 1, dx * dy).reshape((dy, dx))
    Slin, dq = linearity(S, linearity_file, origin=(xmin, ymin))
    Sfwd, exflag = invlinearity(Slin, linearity_file, origin=(xmin, ymin))

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


def test_IL(linearity_file, gain_file, ipc_file):
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

    ILTEST = IL(linearity_file, gain_file, ipc_file)
    n = 4088
    NE = np.zeros((n, n), dtype=np.float32)
    ymin = 260
    ymax = 262
    xmin = 140
    xmax = 143
    print(ILTEST.apply(NE, electrons=True, electrons_out=False)[ymin:ymax, xmin:xmax])
    NE[::3, ::3] = 2.0e3
    print(ILTEST.apply(NE, electrons=True, electrons_out=False)[ymin:ymax, xmin:xmax])


if __name__ == "__main__":
    # some tests

    if len(sys.argv) < 3:
        print("call with python linearity.py <linearity_file> <gain_file> [<ipc_file>].")
        exit()

    test__lin()
    test_lin_ilin(sys.argv[1])
    if len(sys.argv) < 4:
        test_IL(sys.argv[1], sys.argv[2], None)
    else:
        test_IL(sys.argv[1], sys.argv[2], sys.argv[3])
