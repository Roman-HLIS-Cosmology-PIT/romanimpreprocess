import numpy as np
import asdf

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
