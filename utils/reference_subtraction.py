import numpy as np
import sys
import asdf
from roman_datamodels.dqflags import pixel

def ref_subtraction_channel(image, channel_start=0, channel_end=128, use_ref_channel=False):
    """
    Performs a simple channel-wise reference pixel subtraction on the slopes image.
    Calculates a linear fit to the median pixel values at the top and bottom of each channel,
    and subtracts the fitted line from each column in the channel.
    
    Parameters:
    image: a 2D numpy array representing the slopes image.
    channel_start: The starting index for the first channel (default is 0).
    channel_end: The ending index for the first channel (default is 128).
    use_ref_channel: bool, whether to use "channel 33"

    Output:
    image:  2D numpy array with the reference pixel values subtracted from each column in each channel.
    The image is expected to have 33 channels, each with 128 columns.
    """
    # Define beginning and ending indices for the first channel as an initial starting point
    channel_start = channel_start
    channel_end = channel_end
    if use_ref_channel: n_channels=33
    else: n_channels=32

    # Vertical reference pixel subtraction
    for channel in range(0, n_channels):

        ch = image[:, channel_start:channel_end]

        bottom_med = np.median(ch[0:4, :])
        top_med = np.median(ch[4092:4096, :])

        # Do a least squares fit solution to fit a line to the top and bottom medians
        points = [(1.5, bottom_med), (4093.5, top_med)]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m_cor, c_cor = np.linalg.lstsq(A, y_coords,rcond=None)[0]

        # Use the lstsq fit from above to compute a reference pixel median value to be subtracted from each col
        def cor_func(j):
            I_el = m_cor * j + c_cor
            return I_el
        # For all the columns, compute the value from the lstsq fit, and subtract it from all pixels in that col
        for j in range(0, 4096):
            I_el = cor_func(j)
            image[j, channel_start:channel_end] = image[j, channel_start:channel_end] - I_el

        # Iterate
        channel_start = channel_start + 128
        channel_end = channel_end + 128

    return image

def ref_subtraction_row(image, use_ref_channel=False):
    """
    Performs a simple row-wise reference pixel subtraction on the slopes image.
    Fits active-region median as a funciton of reference-region median, subtracts the
    fitted median from each row.

    Parameters:
    image: a 2D numpy array representing the slopes image.
    use_ref_channel: bool, whether to use "channel 33" for fitting
    
    Output: 
    image: a 2D numpy array with the reference pixel values subtracted from each row."""

    sci_medians = []
    ref_medians = []
    for row in range(0, 4096):
        sci_medians.append(np.median(image[row, 4:4088]))
        if use_ref_channel:
            ref_medians.append(np.median(image[row, 4096:4224]))
        else:
            ref_medians.append(np.median(np.hstack((image[row, 0:4], image[row,4088:4096]))))

    m_med, b_med = np.polyfit(ref_medians, sci_medians, 1) 

    def med_func(i):
        I_med = m_med * i + b_med
        return I_med

    # Iterate through all rows: compute median value, subtract it across the whole row
    for i in range(0,4096):
        I_med = med_func(ref_medians[i])
        image[i, :] = image[i, :] - I_med

   
    return image
