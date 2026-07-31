"""Utilities for pulling out the orientation of an image and making FITS data."""

import pathlib

import asdf
import numpy as np

# SCA reference positions
sca_ref_pos = np.array(
    [
        [-0.06784, -0.03653],
        [-0.0678, 0.10972],
        [-0.06769, 0.24053],
        [-0.2034, -0.0636],
        [-0.2035, 0.08296],
        [-0.20338, 0.21345],
        [-0.33864, -0.12921],
        [-0.33894, 0.01811],
        [-0.34003, 0.14753],
        [0.06784, -0.03653],
        [0.0678, 0.10972],
        [0.06769, 0.24053],
        [0.2034, -0.0636],
        [0.2035, 0.08296],
        [0.20338, 0.21345],
        [0.33864, -0.12921],
        [0.33894, 0.01811],
        [0.34003, 0.14753],
    ]
)


def get_orientation(afile):
    """
    Extracts the orientation information from ASDF metadata.

    This isn't for precision applications, it is *only* for use for figuring out
    which SCAs need to be drawn.

    Inputs
    ------
    afile : AsdfFile or str
        An ASDF file populated with Roman metadata in the L1 schema.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ra, dec, pa : float

          The RA, Dec, and PA of the WFI center.

        - ra_sca, dec_sca : np.ndarray of float

          The RA and Dec of the 18 SCA centers (from WFI01..WFI18, length 18).

        All outputs are in degrees.

    """

    degree = np.pi / 180.0  # need this

    # Get the data.
    if isinstance(afile, (str | pathlib.Path)):
        with asdf.open(afile) as _a:
            meta = _a["roman"]["meta"]
    else:
        meta = afile["roman"]["meta"]
    # position
    dec_ref = meta["wcsinfo"]["dec_ref"] * degree
    ra_ref = meta["wcsinfo"]["ra_ref"] * degree
    roll_ref = meta["wcsinfo"]["roll_ref"] * degree
    # aberration
    try:
        scale_factor = meta["velocity_aberration"]["scale_factor"]
    except KeyError:
        scale_factor = 1.0

    # Rotation matrix *from* field angles *to* J2000

    # INT +X = FPA +X
    # INT +Z = telescope boresight
    offset = 0.496 * degree

    # BST +Z = telescope boresight
    # BST -X = toward NCP
    roll = -150.0 * degree + roll_ref

    # J2000 <- BST <- INT <- FPA
    rmat = (
        np.array([[np.cos(ra_ref), -np.sin(ra_ref), 0], [np.sin(ra_ref), np.cos(ra_ref), 0], [0, 0, 1]])
        @ np.array([[np.sin(dec_ref), 0, np.cos(dec_ref)], [0, 1, 0], [-np.cos(dec_ref), 0, np.sin(dec_ref)]])
        @ np.array([[np.cos(roll), np.sin(roll), 0], [-np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, -np.cos(offset), np.sin(offset)], [0, -np.sin(offset), -np.cos(offset)]])
    )

    # field center
    ra = np.arctan2(rmat[1, 2], rmat[0, 2]) / degree + 180.0
    dec = np.arctan2(-rmat[2, 2], np.hypot(rmat[0, 2], rmat[1, 2])) / degree

    # SCA center positions
    coords = np.zeros((3, 19))  # add one position for the FPA +Y direction
    coords[:2, :18] = sca_ref_pos.T * degree / scale_factor
    coords[:2, :18] *= np.sinc(np.hypot(coords[0, :18], coords[1, :18]) / np.pi)[None, :]
    coords[2, :18] = -np.sqrt(1.0 - coords[0, :18] ** 2 - coords[1, :18] ** 2)
    coords[1, 18] = 1.0
    coords_j2000 = rmat @ coords
    # get the SCAs --- only the first 18 positions
    ra_sca = np.arctan2(-coords_j2000[1], -coords_j2000[0])[:-1] / degree + 180.0
    dec_sca = np.arctan2(coords_j2000[2], np.hypot(coords_j2000[0], coords_j2000[1]))[:-1] / degree

    # now get a position 90 degrees from the WFI center, in the "North" direction.
    v_in_wfi_coords = rmat.T @ np.array(
        [-np.sin(dec_ref) * np.cos(ra_ref), -np.sin(dec_ref) * np.sin(ra_ref), np.cos(dec_ref)]
    )
    pa = np.arctan2(-v_in_wfi_coords[0], -v_in_wfi_coords[1]) / degree + 180.0

    return {"ra": ra, "dec": dec, "pa": pa, "ra_sca": ra_sca, "dec_sca": dec_sca}
