import galsim
import gwcs
import numpy as np
from astropy.wcs import WCS


def pixelarea(inwcs, N=4088):
    """Generates an (N,N)-shaped array of the solid angles of the pixels."""

    sp = np.linspace(-1, N + 1, N + 2)
    xx, yy = np.meshgrid(sp, sp)
    deg = np.pi / 180.0

    # get the world coordinates
    if isinstance(inwcs, WCS):
        # try astropy WCS
        world_coords = inwcs.wcs_pix2world(np.vstack((xx.ravel(), yy.ravel())).T, 0)
        ra = world_coords[:, 0] * deg
        dec = world_coords[:, 1] * deg
    elif isinstance(inwcs, galsim.BaseWCS):
        # try galsim WCS
        ra, dec = inwcs.toWorld(xx.ravel(), yy.ravel(), units=galsim.degrees)
        ra *= deg
        dec *= deg
    elif isinstance(inwcs, gwcs.wcs.WCS):
        # try gwcs WCS
        ra, dec = inwcs(xx.ravel(), yy.ravel(), with_bounding_box=False)
        ra *= deg
        dec *= deg
    else:
        raise ValueError("Unrecognized WCS type")

    # now construct areas from these.
    # a solution is to re-project in equal-area coordinates
    # but there is a singularity at the pole, so we can choose our pole to be in the same
    # hemisphere as the beginning of the array.
    theta = np.pi / 2.0 + dec
    if dec[0] > 0:
        theta = np.pi / 2.0 - dec

    # projected coordinates
    # u = 2 sin (theta/2) cos ra
    # v = 2 sin (theta/2) sin ra
    rho = 2.0 * np.sin(theta / 2.0)
    u = (rho * np.cos(ra)).reshape((N + 2, N + 2))
    v = (rho * np.sin(ra)).reshape((N + 2, N + 2))
    del rho

    # now the Jacobian terms
    J11 = (u[1:-1, 2:] - u[1:-1, :-2]) / 2.0
    J12 = (u[2:, 1:-1] - u[:-2, 1:-1]) / 2.0
    J21 = (v[1:-1, 2:] - v[1:-1, :-2]) / 2.0
    J22 = (v[2:, 1:-1] - v[:-2, 1:-1]) / 2.0

    # pixel area in sr
    Area = np.abs(J11 * J22 - J21 * J12)
    return Area
