import numpy as np
import sys
from astropy import units as u
from astropy.io import fits
from scipy.signal import convolve
import asdf
import yaml
from copy import deepcopy

import roman_datamodels
import roman_datamodels.maker_utils as maker_utils
from roman_datamodels.dqflags import pixel
from romanisim import parameters, image as rimage, persistence as rip, __version__ as rstversion, wcs as riwcs

# not actually doing a simulation but needed to pass around the WCS types
import galsim

# local imports
from . import oututils
from ..utils import bitutils, ipc_linearity, fitting, flatutils, coordutils, maskhandling, processlog, reference_subtraction
from .. import pars

# stcal imports
from stcal.saturation.saturation import flag_saturated_pixels

### function definitions below here

def wcs_from_config(config):
    """Gets a WCS object from the configuration."""

    if 'FITSWCS' in config:
        with open(config['FITSWCS']) as f:
            return fits.Header.fromstring(f.read())

    # if no WCS was found, just return None (we'll deal with this later)
    return None

def initializationstep(config, caldir, mylog):
    """Initialization step. Returns:

    data : 3D numpy array, data
    rdq : 3D numpy array, flags (ramp data quality)
    pdq : 2D numpy array, flags (pixel data quality)
    meta : dictionary of assorted other metadata
           (right now: frame_time and read_pattern)
    l1meta : metadata stright from the L1 file
    amp33 : reference output
    """

    with asdf.open(config['IN']) as f:
        data = np.copy(f['roman']['data'].astype(np.float32))
        amp33 = np.copy(f['roman']['amp33'].astype(np.float32))
        rdq = np.zeros(np.shape(data), dtype=np.uint32)

        # guide windows
        # in DCL testing the rows containing the guide windows were affected
        # we expect also that the pixels with IPC coupling to the guide window
        # are affected at some level so I'm flagging those too.
        guide_star = f['roman']['meta']['guide_star']
        xstart = int(guide_star['window_xstart'])
        xstop  = int(guide_star['window_xstop'])
        ystart = int(guide_star['window_ystart'])
        ystop  = int(guide_star['window_ystop'])
        mylog.append('guide window: x={:d}:{:d}, y={:d}:{:d}\n'.format(xstart, xstop, ystart, ystop))
        # if the metadata contain a real window, mask that row
        if xstart>=0 and ystart>=0 and xstop<=4096 and ystop<=4096:
            rdq[:,:,xstart:xstop] |= pixel.GW_AFFECTED_DATA
            # now flag potential IPC
            if xstart>4: xstart -= 1
            if xstop<4092: xstop += 1
            if ystart>4: ystart -= 1
            if ystop<4092: ystop += 1
            rdq[:,ystart:ystop,xstart:xstop] |= pixel.GW_AFFECTED_DATA

        # pull out metadata that we want later
        meta = {
               'frame_time': f['roman']['meta']['exposure']['frame_time'],
               'read_pattern': f['roman']['meta']['exposure']['read_pattern']
        }

        # more information
        meta['ngrp'] = len(meta['read_pattern'])
        meta['tbar'] = np.zeros(meta['ngrp'], dtype=np.float32)
        meta['tau'] = np.zeros(meta['ngrp'], dtype=np.float32)
        meta['N'] = np.zeros(meta['ngrp'], dtype=np.int16)
        for i in range(meta['ngrp']):
            # N_i, tbar_i, and tau_i as defined in Casertano et al. 2022
            meta['N'][i] = len(meta['read_pattern'][i])
            t0  = meta['read_pattern'][i][0]
            meta['tbar'][i] = ( t0 + (meta['N'][i]-1)/2. ) * meta['frame_time']
            meta['tau'][i] = ( t0 + (meta['N'][i]-1)*(2*meta['N'][i]-1)/(6.*meta['N'][i]) ) * meta['frame_time']

        l1meta = deepcopy(f['roman']['meta'])

    # mask
    if 'mask' in caldir:
        with asdf.open(caldir['mask']) as m:
            rdq |= m['roman']['dq'][None,:,:]

    # pixel dq
    pdq = np.bitwise_or.reduce(rdq, axis=0)

    return data, rdq, pdq, meta, l1meta, amp33

def saturation_check(data, read_pattern, rdq, pdq, caldir, mylog):
    """Performs a saturation check on the data cube (data) using the calibration files in caldir.
    Information is appended to mylog. The flags rdq and pdq are updated in place.

    This function serves as a wrapper for flag_saturated_pixels (imported from stcal).
    """

    # passing the 0th frame will lead to division by zero, so we avoid this
    # start the saturation check with the s th frame
    s = 0
    if read_pattern[0]==[0]: s=1

    with asdf.open(caldir['saturation']) as f:
        flag_saturated_pixels(
            data[None,s:,:,:], # flag_saturated_pixels expects a 4D array with integrations as the 0-axis
            rdq[None,s:,:,:], # ramp data quality, with only 1 integration, expanded to 4D
            pdq, # 2D pixel, passed through
            f['roman']['data'], # saturation threshold, 2D
            np.copy(f['roman']['dq']), # saturation quality flags
            2**16-1, # maximum of ADC output -- 16 bits
            pixel, # this is the Roman data quality flag array
            n_pix_grow_sat=1, # also flag 1 pixel around each saturated one
            zframe=None,
            read_pattern=read_pattern[s:], # again, this is a list of list of ints
            bias=None
        )

    # backs up 1 frame to be safe since if the non-linearity curve is sharp enough
    # the existing algorithm can fail on a large group
    # important to run this in ascending order
    for i in range(len(read_pattern)-1):
        if len(read_pattern[i])>1:
            rdq[i,:,:] |= rdq[i+1,:,:]&pixel.SATURATED

def subtract_dark_current(data,rdq,pdq,caldir,meta,mylog):
    """Subtracts dark current from a linearized image.

    Inputs:
    data = 3D data cube (in DN_lin, shape ngroup,4096,4096)
    rdq = 3D ramp data quanity (uint32, shape ngroup,4096,4096)
    pdq = 2D pixel data quanity (uint32, shape 4096,4096)
    caldir = calibration dictionary
    meta = metadata
    mylog = log object

    The data, rdq, and pdq are updated in place.

    Returns:
    dcsub = subtracted dark current in DN/s
    """

    with asdf.open(caldir['dark']) as f:
        dcsub = np.copy(f['roman']['dark_slope'])
    ngrp = meta['ngrp']
    for j in range(ngrp):
        data[j,:,:] -= meta['tbar'][j]*dcsub
    return dcsub

def repackage_wcs(thewcs):
    """Packages a WCS to feed to romanisim.
    """

    # make WCS --- a few ways of doing this
    while True:
        wcsobj = None
        class Blank:
            pass

        # first try a FITS header
        if isinstance(thewcs, fits.Header):
            wcsobj = Blank()
            wcsobj.header = Blank()
            wcsobj.header.header = thewcs
            break

        # should work if this is a GalSim WCS
        try:
            header = fits.Header()
            thewcs.writeToFitsHeader(header, galsim.BoundsI(0,4088,0,4088))
            # offset to FITS convention -- this is undone later
            header['CRPIX1'] += 1; header['CRPIX2'] += 1
            wcsobj = Blank()
            wcsobj.header = Blank()
            wcsobj.header.header = header
            warnings.warn('Use of GalSim WCS in calibrate is not fully working yet!')
            break
        except Exception as e:
            if verbose_err:
                print('Tried GalSim, failed')
                print(e)
            wcsobj = None
            raise Exception('Unrecognized WCS')

    return wcsobj

def calibrateimage(config, verbose=True):
    """Main routine to run the specified calibrations from a config file.

    The config is a dictionary intended to be read from a YAML file, though it could also be
    written/edited here.

    """

    # setup
    mylog = processlog.ProcessLog()

    # get an initial WCS (if provided)
    # in some simulations we may need to give this if the input stars themselves are simulated
    thewcs = wcs_from_config(config)
    caldir = config['CALDIR']

    # initialize a data cube and data quality
    data, rdq, pdq, meta, l1meta, amp33 = initializationstep(config, caldir, mylog)
    (ngrp,ny,nx) = np.shape(data)
    nb=meta['nborder']=4
    mylog.append('Initialized data\n')

    # saturation check
    saturation_check(data, meta['read_pattern'], rdq, pdq, caldir, mylog)
    mylog.append('Saturation check complete\n')

    # reference pixel correction -- right now using a 5-pixel filter of the left & right ref pixels
    # and the top & bottom pixel subtraction functions from Laliotis et al. (2024)
    # **This is a placeholder until:
    #  - amp33 to be implemented (currently the simulation leaves it blank)
    #  - improved reference pixel correction from GSFC group should be available
    with asdf.open(caldir['dark']) as f:
        rsub = np.zeros((ngrp,4096), dtype=np.float32)
        for j in range(ngrp):
            image = np.zeros((4096,4224),dtype=np.float32)
            image[:,:4096] = data[j,:,:] - f['roman']['data'][j,:,:]
            with asdf.open(caldir['read']) as fr:
                if 'amp33' in fr['roman']:
                    image[:,-128:] = amp33[j,:,:] - fr['roman']['amp33']['med']
                    image[:,-128:] -= np.median(image[:,-128:])
            #rsub[j,:] = y = np.median(np.roll(image[:,:4096],4,axis=1)[:,:8],axis=1)
            #ksm = 2
            #y = convolve(np.pad(y,ksm,mode='edge'), np.ones(2*ksm+1)/(2*ksm+1), mode='valid', method='direct')
            #image -= y[:,None]
            image = reference_subtraction.ref_subtraction_row(image)
            image = reference_subtraction.ref_subtraction_channel(image)
            data[j,:,:] = image[:,:4096] + f['roman']['data'][j,:,:]

    # bias correction
    if 'biascorr' in caldir:
        with asdf.open(caldir['biascorr']) as f:
            data[:,nb:-nb,nb:-nb] -= f['roman']['data']
        mylog.append('Included bias correction\n')
    else:
        mylog.append('Skipping bias correction\n')

    # linearity correxction
    # ** right now applies the linearity to a group average, which isn't strictly correct **
    # ** will fix this in a future upgrade! **
    data,dq_lin = ipc_linearity.multilin(data,
        caldir['linearitylegendre'],                  # the linearity cube
        do_not_flag_first=meta['read_pattern'][0]==0, # don't flag the first read for being off scale if it is the reset
        attempt_corr= ~rdq & pixel.SATURATED          # don't flag saturated pixels as having a bad linearity correction
    )
    if len(np.shape(dq_lin))==2:
        rdq |= dq_lin[None,:,:]
    else:
        rdq |= dq_lin
    del dq_lin # we have everything we need
    mylog.append('Linearity correction complete\n')
    # now data is in linearized DN, floating point

    # subtract out dark current
    # dcsub is the dark current that was subtracted --- data is updated in place
    dcsub = subtract_dark_current(data,rdq,pdq,caldir,meta,mylog)
    mylog.append('Dark current subtracted')

    # IPC correction
    if 'ipc4d' in caldir:
        ipc_linearity.correct_cube(data,caldir['ipc4d'],mylog,gain_file=caldir['gain'])
    else:
        mylog.append('skipping IPC correction\n')

    # ramp fitting
    u_ = 0.4/1.8/7.**2 # right now force the fitting to 0.4 DN/s, g=1.8 e/DN, sigma_read=7 DN
    meta['K'] = fitting.construct_weights(u_, meta, exclude_first=True)
    mylog.append('\n\nRamp fit optimized for u = {:11.5E} s**-1\n'.format(u_))
    mylog.append('weights = {}\n'.format(meta['K']))
    if 'JUMP_DETECT_PARS' in config: meta['jump_detect_pars'] = config['JUMP_DETECT_PARS']
    slope, slope_err_read, slope_err_poisson = fitting.ramp_fit(data, rdq, pdq, meta, caldir, mylog, exclude_first=True)

    # apply flat field
    flat = flatutils.get_flat(caldir, meta, pdq)
    # this is the ratio of the true pixel area to the reference area (0.11 arcsec)^2
    AreaFactor = coordutils.pixelarea(riwcs.convert_wcs_to_gwcs(repackage_wcs(thewcs)), N=np.shape(slope)[-1]) / pars.Omega_ideal
    print(AreaFactor[::1024,::1024])
    flat = (flat/AreaFactor).astype(np.float32)
    mylog.append('acquired flat field\n')
    for p in [1,2,5,10,25,50,75,90,95,98,99]:
        mylog.append(' {:2d}%ile = {:6.4f},'.format(p, np.percentile(flat,p)))
    mylog.append('\n')
    slope /= flat
    slope_err_read /= flat
    slope_err_poisson /= flat

    # need the median gain to send to a file
    with asdf.open(caldir['gain']) as g_: medgain = np.median(g_['roman']['data'])
    mylog.append('median gain = {:8.5f} e/DN\n'.format(medgain))

    # blank persistence object right now
    persistence = rip.Persistence()

    im2, extras2 = rimage.make_asdf(
        slope[nb:-nb,nb:-nb]*u.DN/u.s, (slope_err_read[nb:-nb,nb:-nb]*u.DN/u.s)**2, (slope_err_poisson[nb:-nb,nb:-nb]*u.DN/u.s)**2,
        metadata=l1meta, persistence=persistence,
        dq=pdq[nb:-nb,nb:-nb], imwcs=repackage_wcs(thewcs), gain=medgain)


    oututils.add_in_ref_data(im2, config['IN'], rdq, pdq)

    # Create metadata for simulation parameter
    romanisimdict2 = {'version': rstversion}
    romanisimdict2.update(**extras2)

    # Write file
    af2 = asdf.AsdfFile()
    af2.tree = {'roman': im2, 'romanisim': romanisimdict2}
    af2.write_to(open(config['OUT'], 'wb'))

    if 'FITSOUT' in config:
        if config['FITSOUT']:
           good = ~ maskhandling.PixelMask1.build(im2['dq']) # this is one choice

           # note we accept saturated pixels in this step
           fits.HDUList([fits.PrimaryHDU(im2['data']),
                         fits.ImageHDU(im2['dq']),
                         fits.ImageHDU(np.where(good, im2['data'], -1000))]
                        ).writeto(config['OUT'][:-5]+'_asdf_to.fits', overwrite=True)

    print(mylog.output)

    return
    # test stuff below here -- shouldn't be executed because of the return
    #print(pdq[:40:4,:40:4])
    #fits.PrimaryHDU(bitutils.convert_uint32_to_bits(rdq[-1,:,:])).writeto('rdq.fits', overwrite=True)
    #fits.PrimaryHDU(bitutils.convert_uint32_to_bits(pdq)).writeto('pdq.fits', overwrite=True)
    #fits.PrimaryHDU(slope).writeto('slope.fits', overwrite=True)
    #slopemask = pdq & (pixel.JUMP_DET | pixel.DO_NOT_USE | pixel.NO_LIN_CORR | pixel.HOT) != 0
    #fits.PrimaryHDU(np.where(np.logical_not(slopemask),slope,-1000.)).writeto('slopemask.fits', overwrite=True)

if __name__=="__main__":

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    calibrateimage(config)
