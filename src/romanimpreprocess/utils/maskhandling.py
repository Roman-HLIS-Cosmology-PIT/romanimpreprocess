import asdf
import numpy as np
from astropy.io import fits
from roman_datamodels.dqflags import pixel
from scipy.signal import convolve

"""Tools for building a mask by growing the input bitmask by
different amounts.
"""


class CombinedMask:
    """Class to generate a boolean mask.

    Attributes:
    array : dictionary of how much to grow each flag.

    Methods:
    __init__ : constructor
    build : make boolean mask from unit32 array
    convert_file : stand-alone function to make a mask from a file

    Examples:
    To take a dq array, and flag any pixel with 'gw_affected_data'
    and the cardinal-nearest neigbors if 'jump_det' is set:

    myMaskFunc = CombinedMask({'jump_det': 5, 'gw_affected_data': 1})
    mymask = myMaskFunc.build(dq)

    (Note that capitalization is automatic so this function is not case-sensitive.)

    Options for growing are specified by the number of pixels affected:
     1 = that pixel
     5 = cardinal nearest neighbors
     9 = 3x3 block
    """

    # Kernel dictionary is a class variable
    kerneldict = {
        5: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.int16),
        9: np.ones((3, 3), dtype=np.int16),
        25: np.ones((5, 5), dtype=np.int16),
    }

    def __init__(self, maskdict):
        """Constructor function from a dictionary."""
        self.array = np.zeros(32, dtype=np.uint8)
        for d in maskdict:
            if isinstance(d, int):
                whichbit = d
            if isinstance(d, str):
                e = getattr(pixel, d.upper())
                whichbit = 0
                for x in range(32):
                    if e >> x == 1:
                        whichbit = x

            self.array[whichbit] = int(maskdict[d])

    def build(self, dq):
        """Make a boolean mask from a dq array (True = masked)."""

        (ny, nx) = np.shape(dq)
        mask = np.zeros((ny, nx), dtype=bool)

        # loop over each bit
        for whichbit in range(32):
            if self.array[whichbit] > 0:
                layer = np.where(np.uint32(2**whichbit) & dq != 0, 1, 0).astype(np.int16)

                # now different types of growing masks. 1 is a simple copy, >1 leads to a convolution
                if self.array[whichbit] == 1:
                    mask |= layer >= 1
                else:
                    mask |= (
                        convolve(
                            2 * layer, self.kerneldict[self.array[whichbit]], mode="same", method="direct"
                        )
                        >= 1
                    )

        return mask

    def convert_file(self, file_in, file_mask):
        """Makes a file_mask from the file_in L2 file.

        If .asdf is requested, simply writes the boolean array.
        If .fits is requested, makes a masked image (HDU0, for display purposes) and an int8 version (HDU1).
        """

        with asdf.open(file_in) as f_in:
            locmask = self.build(f_in["roman"]["dq"])
            if file_mask[-5:] == ".asdf":
                asdf.AsdfFile({"mask": locmask}).write_to(file_mask)
            elif file_mask[-5:] == ".fits":
                h1 = fits.PrimaryHDU(np.where(locmask, -1000.0, f_in["roman"]["data"]).astype(np.float32))
                h2 = fits.ImageHDU(np.where(locmask, 1, 0).astype(np.int8))
                h2.header["EXTNAME"] = "MASK"
                fits.HDUList([h1, h2]).writeto(file_mask, overwrite=True)


# Some specific choices you may want

PixelMask1 = CombinedMask(
    {
        "DO_NOT_USE": 1,
        "JUMP_DET": 5,
        "DROPOUT": 25,
        "GW_AFFECTED_DATA": 1,
        "PERSISTENCE": 1,
        "AD_FLOOR": 5,
        "UNRELIABLE_ERROR": 1,
        "NON_SCIENCE": 1,
        "DEAD": 9,
        "HOT": 9,
        "WARM": 1,
        "LOW_QE": 9,
        "TELEGRAPH": 1,
        "NO_FLAT_FIELD": 9,
        "NO_GAIN_VALUE": 9,
        "NO_LIN_CORR": 9,
        "NO_SAT_CHECK": 9,
        "UNRELIABLE_BIAS": 1,
        "UNRELIABLE_DARK": 9,
        "UNRELIABLE_SLOPE": 9,
        "UNRELIABLE_FLAT": 9,
        "UNRELIABLE_RESET": 9,
        "OTHER_BAD_PIXEL": 9,
    }
)
