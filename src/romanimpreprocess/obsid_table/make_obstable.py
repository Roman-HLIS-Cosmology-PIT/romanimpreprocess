import copy
import sys

import asdf
import numpy as np
import roman_datamodels as rdm
from asdf.yamlutil import tagged_tree_to_custom_tree
from astropy.time import Time


def make_obstable_row(wcs_file):
    """Builds a dictionary from a WCS file.

    Parameters
    ----------
    wcs_file : str
        The path to the WCS file to read.

    Returns
    -------
    dict
        The dictionary of data for this file that we're going to put in the observation table.

    """

    deg = np.pi / 180.0  # lots of conversions with degrees

    filter_mapping = {
        "F062": "R062",
        "F087": "Z087",
        "F106": "Y106",
        "F129": "J129",
        "F158": "H158",
        "F184": "F184",
        "F213": "K213",
        "F146": "W146",
        "P127": "PRSM",
        "G150": "GRSM",
        "DARK": "DARK",
    }

    with asdf.open(wcs_file) as af:
        awcs = copy.deepcopy(af["roman"]["wcs_l2"])
        with asdf.AsdfFile() as ag:
            g = tagged_tree_to_custom_tree(
                awcs, ag
            )  # open the gwcs file # now that this is a gwcs we can react with it normally
        f = rdm.open(af).copy()

    # save the meta data
    date = f.meta.file_date
    exptime = f.meta.exposure.exposure_time
    filter = f.meta.instrument.optical_element
    filename = f.meta.filename
    program = f.meta.observation.program
    obsID = f.meta.observation.observation_id
    # other SOC meta data?

    # convert the date to MJD
    date_MJD = Time(date, format="iso", scale="utc").mjd

    # find the ra and dec values of the WFI center, C
    # define field positions (fixed to WFI01) in degrees
    # A defined as (0,0) and B defined as (0,4087)
    XANA = -0.00520 * deg
    YANA = 0.02475 * deg
    XANB = -0.00522 * deg
    YANB = -0.09810 * deg

    # find the angles between
    AC = np.sqrt(XANA**2 + YANA**2)
    BC = np.sqrt(XANB**2 + YANB**2)
    AB = np.sqrt((XANA - XANB) ** 2 + (YANA - YANB) ** 2)

    # define the constants used to find C
    # these formulae were unstable for small angles so I re-wrote them to not have cancellations
    # when cos(angle) is close to 1. The identity is 1 - cos theta = 2 sin^2 (theta/2)
    # in the below, 'omc' = 'one minus cosine'
    omcAB = 2 * np.sin(AB / 2) ** 2
    omcAC = 2 * np.sin(AC / 2) ** 2
    omcBC = 2 * np.sin(BC / 2) ** 2
    c1 = (-omcAC + omcAB + omcBC - omcAB * omcBC) / (np.sin(AB)) ** 2
    c2 = (-omcBC + omcAB + omcAC - omcAB * omcAC) / (np.sin(AB)) ** 2
    c3 = np.sqrt(1.0 - (np.sin(AB)) ** -2 * (2 * omcAB * np.cos(AC) * np.cos(BC) + (omcAC - omcBC) ** 2))
    #
    # previous 'direct' implementation
    # c1 = (np.cos(AC) - np.cos(AB)*np.cos(BC)) / (np.sin(AB))**2
    # c2 = (np.cos(BC) - np.cos(AB)*np.cos(AC)) / (np.sin(AB))**2
    # c3 = np.sqrt(1 - (np.sin(AB))**-2 * (np.cos(AC)**2 + np.cos(BC)**2
    #      - 2*np.cos(AB)*np.cos(AC)*np.cos(BC)))

    # get A and B in wcs coordinates (RA, DEC)
    ra_A, dec_A = g(0, 0)
    ra_B, dec_B = g(0, 4087)

    print(ra_A, dec_A, ra_B, dec_B)

    ra_A *= deg
    dec_A *= deg
    ra_B *= deg
    dec_B *= deg

    # get the vectors for A, B, P (where P is defined as being perpendicular to both)
    A = np.array([np.cos(dec_A) * np.cos(ra_A), np.cos(dec_A) * np.sin(ra_A), np.sin(dec_A)])
    B = np.array([np.cos(dec_B) * np.cos(ra_B), np.cos(dec_B) * np.sin(ra_B), np.sin(dec_B)])
    P = np.cross(B, A) / np.abs(np.cross(B, A))

    # using the constants and the vectors, calculate C
    C = c1 * A + c2 * B + c3 * P
    # calculate the ra and dec for C
    ra_C = np.arctan2(C[1], C[0])
    dec_C = np.arctan2(C[2], np.sqrt(C[0] ** 2 + C[1] ** 2))

    # calculate the pa
    N = np.array([-np.sin(dec_C) * np.cos(ra_C), -np.sin(dec_C) * np.sin(ra_C), np.cos(dec_C)])
    E = np.array([-np.sin(ra_C), np.cos(ra_C), 0])
    PA = np.arctan2(np.dot(E, (A - B)), np.dot(N, (A - B)))

    # convert outputs to degrees
    ra_C /= deg
    ra_C = ra_C if ra_C > 0 else ra_C + 360  # put in range
    dec_C /= deg
    PA /= deg
    PA = PA if PA > 0 else PA + 360  # put in range

    table_row = {
        "date": date_MJD,
        "exptime": exptime,
        "ra": ra_C,
        "dec": dec_C,
        "pa": PA,
        "filter": filter_mapping[filter],
        "filename": filename,
        "program": program,
        "OBSID": obsID,
    }

    return table_row


# use case to show that the function itself workds
if __name__ == "__main__":
    # run the function and print
    print(make_obstable_row(sys.argv[1]))
