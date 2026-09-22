'''
Functions to correct BFE in H4RG images using a modified version of the 
scalar kernel correction in Coulton et al. (2018). 

TO-DO:
- update correct_cube_frame_difference for unequally spaced frames (resultants)
'''

import numpy as np

from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import sigma_clipped_stats


def make_bfe_kernel(scaling, k00, k10, k11, k20, k21, k22):
    '''
    Construct a symmetric 5x5 BFE kernel.

    Parameters
    ----------
    scaling : float
        Overall multiplicative scaling applied to the kernel.
    k00, k10, k11, k20, k21, k22 : float
        BFE kernel coefficients, where the indices indicate the
        pixel offset from the kernel center.

    Returns
    -------
    bfe_kernel : ndarray
        Symmetric 5x5 BFE kernel multiplied by `scaling`.
    '''

    bfe_kernel = np.array([[k22, k21, k20, k21, k22],
                           [k21, k11, k10, k11, k21],
                           [k20, k10, k00, k10, k20],
                           [k21, k11, k10, k11, k21],
                           [k22, k21, k20, k21, k22]])
    return bfe_kernel*scaling


def correct_cube_frame_difference(image_cube, kernel, dt=3., gain=1., 
                                  threshold=1., max_iters=10, use_prev_corr=False):
    '''
    Correct BFE using a modified version of the Coulton+2018 (DOI: 10.3847/1538-3881/aac08d) method.
    Correction is calculated from differences between consecutive frames, 
    then integrate the corrected fluxes to reconstruct the image cube.

    Parameters
    ----------
    image_cube : array-like
        Input cube with shape (num_frames, Ny, Nx), in DN.
        Frames are assumed to be equally spaced in time. 
    kernel : ndarray
        BFE kernel.
    dt : float, optional
        Time between frames. Default is 3.
    gain : float, optional
        Detector gain in electrons/DN. Default is 1.
    threshold : float, optional
        Convergence threshold for the summed change in corrected charge.
        Default is 1.
    max_iters : int, optional
        Maximum number of iterations per frame pair. Default is 10.
    use_prev_corr : bool, optional
        If True, initialize each flux using the correction from the
        preceding pair. This can slightly reduce the number of iterations
        required for convergence. Default is False.

    Returns
    -------
    corr_cube : array-like
        BFE-corrected image cube in DN, with the original mask restored.
    '''

    nframes = image_cube.shape[0]

    # Interpolate masked pixels using Gaussian kernel
    g_kernel = Gaussian2DKernel(1,1)
    reconstructed_cube = image_cube.copy() * gain 
    for i, frame in enumerate(reconstructed_cube):
        frame[frame.mask] = np.nan
        reconstructed_cube[i] = interpolate_replace_nans(frame.data, g_kernel)
            
    corr_cube = reconstructed_cube.copy()

    corr = np.zeros_like(image_cube[0])

    for f in range(1,nframes):
        print('Starting frame {}'.format(f))
        frame0 = reconstructed_cube[f-1].copy() 
        frame1 = reconstructed_cube[f].copy() 

        # Frame-difference flux in electrons per unit time.
        flux = (frame1 - frame0) / dt 

        # Initialize the correction, which is zero if use_prev_corr==False
        # or is the correction derived from the previous frame pair if True
        if not use_prev_corr:
            corr = np.zeros_like(frame0)
        temp_flux = flux.copy() + corr

        prev_flux = np.zeros_like(frame0)

        # Derivatives of the kernel-convolved accumulated charge in frame0.
        # Can be calculated outside of the iterative while loop since 
        # only the flux between frames is modified per iteration. 
        _, fill, _ = sigma_clipped_stats(frame0)
        conv = convolve(frame0, kernel, boundary='fill', nan_treatment='fill', fill_value=fill,  normalize_kernel=False)
        gradConv = np.gradient(conv)
        gradConv20 = np.gradient(gradConv[0], axis=0)
        gradConv21 = np.gradient(gradConv[1], axis=1)

        iter = 0
        condition = np.sum( np.abs(prev_flux-temp_flux)*dt )

        while (iter<max_iters) & (condition > threshold):

            print('  Working on iteration {}\n    convergence condition is {}'.format(iter, np.max(condition)) ) 
            prev_flux = temp_flux.copy()

            # BFE correction from first and second derivative terms
            gradFlux = np.gradient(prev_flux)
            first = (gradFlux[0]*gradConv[0] + gradFlux[1]*gradConv[1])
            second = prev_flux*(gradConv20 + gradConv21)
            corr = (first + second) 

            temp_flux = flux.copy() + corr
            temp_flux.mask = flux.mask

            iter+=1

            # Calculate total change in charge between correction iterations
            # for convergence condition check 
            condition = np.sum( np.abs(prev_flux-temp_flux)*dt )

        print('  Finished after {} iterations, convergence condition was {}'.format(iter, condition) ) 

        # Integrate the corrected frame-to-frame flux.
        corr_cube[f] = corr_cube[f-1] + temp_flux * dt


    # Restore the original mask and convert electrons back to DN.
    corr_cube.mask = image_cube.mask

    return corr_cube  / gain



def correct_cube_per_frame(image_cube, kernel, gain=1., 
                           threshold=1., max_iters=10):
    '''
    Correct BFE using Coulton+2018 (DOI: 10.3847/1538-3881/aac08d) method, 
    applied to each frame in an image cube 

    Parameters
    ----------
    image_cube : array-like
        Input cube with shape (num_frames, Ny, Nx), in DN.
    kernel : ndarray
        BFE kernel.
    gain : float, optional
        Detector gain in electrons/DN. Default is 1.
    threshold : float, optional
        Convergence threshold for the summed change in corrected charge.
        Default is 1.
    max_iters : int, optional
        Maximum number of iterations per frame pair. Default is 10.

    Returns
    -------
    corr_cube : array-like
        BFE-corrected image cube in DN, with the original mask restored.
    '''

    nframes = image_cube.shape[0]

    # Interpolate masked pixels using Gaussian kernel
    g_kernel = Gaussian2DKernel(1,1)
    reconstructed_cube = image_cube.copy() * gain 
    for frame in reconstructed_cube:
        frame[frame.mask] = np.nan
        frame = interpolate_replace_nans(frame.data, g_kernel)
            
    corr_cube = reconstructed_cube.copy()

    corr = np.zeros_like(image_cube[0])

    for f in range(nframes):
        print('Starting frame {}'.format(f))

        frame_reconstructed = reconstructed_cube[f].copy() 

        corr = np.zeros_like(frame_reconstructed)

        temp_image = frame_reconstructed.copy()
        prev_image = np.zeros_like(temp_image)


        iter = 0
        condition = np.sum(np.abs(prev_image-temp_image)) 

        while (iter<max_iters) & (condition > threshold):
            print('Working on iteration {}, convergence condition is {}'.format(iter, np.sum(np.abs(prev_image-temp_image))) ) 
            prev_image = temp_image.copy()

            _, fill, _ = sigma_clipped_stats(prev_image)
            
            conv = convolve(prev_image, kernel, boundary='fill', nan_treatment='fill',fill_value=fill, normalize_kernel=False)

            # First derivative term
            gradFrame = np.gradient(frame_reconstructed)
            gradConv = np.gradient(conv)
            first = (gradFrame[0]*gradConv[0] + gradFrame[1]*gradConv[1])

            # Second derivative term
            gradConv20 = np.gradient(gradConv[0], axis=0)
            gradConv21 = np.gradient(gradConv[1], axis=1)
            second = frame_reconstructed*(gradConv20 + gradConv21)

            # BFE correction from first and second derivative terms
            corr = 0.5*(first + second)

            temp_image = frame_reconstructed + corr

            iter+=1

            # Calculate total change in charge between correction iterations
            # for convergence condition check 
            condition = np.sum(np.abs(prev_image-temp_image))



        print('Finished after {} iterations, convergence condition was {}'.format(iter, np.sum(np.abs(prev_image-temp_image))) ) 

        corr_cube[f] += corr

    # Restore the original mask and convert electrons back to DN.
    corr_cube.mask = image_cube.mask

    return corr_cube  / gain
