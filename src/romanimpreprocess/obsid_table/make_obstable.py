import asdf
import gwcs
import numpy as np
import roman_datamodels as rdm
from asdf.yamlutil import tagged_tree_to_custom_tree
from astropy.time import Time

import copy
import sys


def make_obstable_row(wcs_file):
    with asdf.open(wcs_file) as af:
        awcs = copy.deepcopy(af["roman"]["wcs_l2"])
        with asdf.AsdfFile() as ag:
            g = tagged_tree_to_custom_tree(awcs, ag) # open the gwcs file # now that this is a gwcs we can react with it normally
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
    XANA = -0.00520
    YANA = 0.02475
    XANB = -0.00522
    YANB = -0.09810

    # find the angles between 
    AC = np.sqrt(XANA**2 + YANA**2)
    BC = np.sqrt(XANB**2 + YANB**2)
    AB = np.sqrt((XANA - XANB)**2 + (YANA - YANB)**2)

    # define the constants used to find C 
    c1 = (np.cos(AC) - np.cos(AB)*np.cos(BC)) / (np.sin(AB))**2
    c2 = (np.cos(BC) - np.cos(AB)*np.cos(AC)) / (np.sin(AB))**2
    c3 = np.sqrt(1 - (np.sin(AB))**-2 * (np.cos(AC)**2 + np.cos(BC)**2 - 2*np.cos(AB)*np.cos(AC)*np.cos(BC)))

    # get A and B in wcs coordinates (RA, DEC)
    ra_A, dec_A = g(0,0)
    ra_B, dec_B = g(0,4087)

    # get the vectors for A, B, P (where P is defined as being perpendicular to both)
    A = np.array([np.cos(dec_A)*np.cos(ra_A), np.cos(dec_A)*np.sin(ra_A), np.sin(dec_A)])
    B = np.array([np.cos(dec_B)*np.cos(ra_B), np.cos(dec_B)*np.sin(ra_B), np.sin(dec_B)])
    P = np.cross(B,A) / np.abs(np.cross(B,A))

    # using the constants and the vectors, calculate C 
    C = c1*A + c2*B + c3*P
    # calculate the ra and dec for C
    ra_C = np.arctan2(C[1], C[0])
    dec_C = np.arctan2(C[2], np.sqrt(C[0]**2 + C[1]**2))

    # calculate the pa
    N = np.array([-np.sin(dec_C)*np.cos(ra_C), -np.sin(dec_C)*np.sin(ra_C), np.cos(dec_C)])
    E = np.array([-np.sin(ra_C), np.cos(ra_C), 0])
    PA = np.arctan2(np.dot(E,(A-B)), np.dot(N, (A-B)))

    table_row = {"date": date_MJD, "exptime": exptime, "ra": ra_C, "dec": dec_C, "pa": PA, "filter": filter, "filename": filename, "program": program, "OBSID": obsID}

    return table_row


# use case to show that the function itself workds
if __name__ == "__main__":
    # run the function and print
    print(make_obstable_row(sys.argv[1])) 

