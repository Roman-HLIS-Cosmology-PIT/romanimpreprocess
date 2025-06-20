import numpy as np
import asdf

import roman_datamodels

def add_in_ref_data(rstruct, infile, rdq, pdq):
    """Adds in reference pixel & output data to the given Roman structure.

    Also populates the quality flags.
    """

    # Reference pixel & output data copied from file
    with asdf.open(infile) as fi:
        print(fi.info(max_rows=None))
        rstruct['amp33'] = np.copy(fi['roman']['amp33'])
        rstruct['border_ref_pix_left'] = np.copy(fi['roman']['data'][:,:,:4].astype(np.float32))
        rstruct['border_ref_pix_right'] = np.copy(fi['roman']['data'][:,:,-4:].astype(np.float32))
        rstruct['border_ref_pix_top'] = np.copy(fi['roman']['data'][:,-4:,:].astype(np.float32))
        rstruct['border_ref_pix_bottom'] = np.copy(fi['roman']['data'][:,:4,:].astype(np.float32))

    # Fill in reference pixel flags
    rstruct['dq_border_ref_pix_left'] = np.copy(pdq[:,:4])
    rstruct['dq_border_ref_pix_right'] = np.copy(pdq[:,-4:])
    rstruct['dq_border_ref_pix_top'] = np.copy(pdq[-4:,:])
    rstruct['dq_border_ref_pix_bottom'] = np.copy(pdq[:4,:])

def update_flags(rstruct, ftype):
    """Updates calibration flags. This routine should be kept current with what each
    type does.
    Most likely, you will call this from gen_cal_image and use that as the type.
    """

    cal = rstruct['meta']['cal_step']
    if ftype.lower()=='gen_cal_image':
        cal['dq_init']       = 'COMPLETE' # Data Quality Initialization Step
        cal['saturation']    = 'COMPLETE' # Saturation Identification Step
        cal['refpix']        = 'COMPLETE' # Reference Pixel Correction Step
        cal['linearity']     = 'COMPLETE' # Classical Linearity Correction Step
        cal['dark']          = 'COMPLETE' # Dark Current Subtraction Step
        cal['ramp_fit']      = 'COMPLETE' # Ramp Fitting Step
        cal['assign_wcs']    = 'COMPLETE' # Assign World Coordinate System (WCS) Step
        cal['flat_field']    = 'COMPLETE' # Flat Field Correction Step

def add_in_provenance(rstruct, ftype):
    """Adds in provenance information to a roman L2 structure."""

    if ftype.lower()=='gen_cal_image':
        rstruct['meta']['calibration_software_name'] = roman_datamodels.stnode.CalibrationSoftwareName('gen_cal_image / HLWAS PIT')
        from ..version import __version__
        rstruct['meta']['calibration_software_version'] = roman_datamodels.stnode.CalibrationSoftwareVersion(str(__version__))
