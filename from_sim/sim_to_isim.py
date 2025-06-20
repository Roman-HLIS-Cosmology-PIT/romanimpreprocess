"""Functions to convert external simulated images to Roman L1/L2-like format.

This works entirely at the single exposure level. Some parts wrap romanisim.
"""

import warnings
import re
import numpy as np
import gwcs
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import asdf
import galsim
from galsim import roman
import sys
import yaml
import copy

import roman_datamodels
import roman_datamodels.maker_utils as maker_utils
from romanisim import parameters, util, wcs, image as rimage, ris_make_utils as ris, persistence as rip, l1 as rstl1, __version__ as rstversion

# local imports
from ..utils.ipc_linearity import IL, ipc_rev
from ..utils.coordutils import pixelarea
from .. import pars

def hdu_sip_hflip(data,header):
    """Horizontal flip of SCA and WCS. Assumes SIP convention."""

    (ny,nx) = np.shape(data)
    data[:,:] = data[:,::-1] # flipping the data is the easy part

    # now flip the WCS
    header['CRPIX1'] = nx+1-header['CRPIX1']
    header['CD1_1'] = -header['CD1_1']
    header['CD2_1'] = -header['CD2_1']
    try:
        # if there is a SIP table, we flip it.
        # for A: the even p's need a sign flip to reverse the direction of the SIP u-axis
        # for B: the odd p's need a sign flip to reverse the direction of the SIP u-axis
        a_order = int(header['A_ORDER'])
        b_order = int(header['B_ORDER'])
        for p in range(0,a_order+1,2):
           for q in range(a_order+1-p):
              keyword = 'A_{:1d}_{:1d}'.format(p,q)
              if keyword in header:
                  header[keyword] = -float(header[keyword])
        for p in range(1,b_order+1,2):
           for q in range(b_order+1-p):
              keyword = 'B_{:1d}_{:1d}'.format(p,q)
              if keyword in header:
                  header[keyword] = -float(header[keyword])
    except:
        print('Exception in SIP table, skipping ...')
        pass

def hdu_sip_vflip(data,header):
    """Vertical flip of SCA and WCS. Assumes SIP convention."""

    (ny,nx) = np.shape(data)
    data[:,:] = data[::-1,:] # flipping the data is the easy part

    # now flip the WCS
    header['CRPIX2'] = ny+1-header['CRPIX2']
    header['CD1_2'] = -header['CD1_2']
    header['CD2_2'] = -header['CD2_2']
    try:
        # if there is a SIP table, we flip it.
        # for A: the odd q's need a sign flip to reverse the direction of the SIP v-axis
        # for B: the even q's need a sign flip to reverse the direction of the SIP v-axis
        a_order = int(header['A_ORDER'])
        b_order = int(header['B_ORDER'])
        for q in range(1,a_order+1,2):
           for p in range(a_order+1-q):
              keyword = 'A_{:1d}_{:1d}'.format(p,q)
              if keyword in header:
                  header[keyword] = -float(header[keyword])
        for q in range(0,b_order+1,2):
           for p in range(b_order+1-q):
              keyword = 'B_{:1d}_{:1d}'.format(p,q)
              if keyword in header:
                  header[keyword] = -float(header[keyword])
    except:
        print('Exception in SIP table, skipping ...')
        pass

def make_l1_fullcal(counts, read_pattern, caldir, rng=None, persistence=None, tstart=None):
    """Make an L1 image with the full calibration information.

    This carries out similar steps to romanisim.l1.make_l1, but
    provides us a bit more control over the settings..

    Parameters
    ----------
    counts : galsim.Image object (mean e per pixel)
    read_pattern : list[list] (MA table)
    caldir: dictionary (with the reference files)
    rng : galsim.BaseDeviate
    persistence : romanisim.persistence.Persistence

    Returns
    -------
    l1, dq : 3D image and quality array
    """

    # generate the reset noise (will complain if rng is None!)
    resetnoise = np.zeros_like(counts.array)
    nb = (8192-np.shape(resetnoise)[-1]//2)%256 # get border size
    galsim.GaussianDeviate(rng).generate(resetnoise)
    with asdf.open(caldir['read']) as f:
        resetnoise *= f['roman']['resetnoise'][nb:-nb,nb:-nb]
    with asdf.open(caldir['gain']) as f:
        resetnoise *= f['roman']['data'][nb:-nb,nb:-nb]
    # now resetnoise is a random image in electrons

    tij = rstl1.read_pattern_to_tij(read_pattern)

    # subtract the apporopriate amount of electrons so that we will be at the
    # zero level (on average) after some dark current
    if 'biascorr' in caldir:
        with asdf.open(caldir['biascorr']) as f:
            tbias = float(f['roman']['t0'])
        with asdf.open(caldir['gain']) as g:
            with asdf.open(caldir['dark']) as f:
                resetnoise -= tbias*f['roman']['dark_slope'][nb:-nb,nb:-nb]/g['roman']['data'][nb:-nb,nb:-nb]

    # this model includes linearity *and* IPC.
    # default application is electrons_in=False, electrons_out=False
    # (the .apply method is called in apportion_counts_to_resultants with
    # electrons_in = True -- this means that electrons go in and raw DN go out,
    # so actually includes the gain as well!)
    e2dn_model = IL(
        caldir['linearitylegendre'],
        caldir['gain'],
        caldir['ipc4d'],
        start_e = resetnoise
    )
    # set the size of the data quality array
    e2dn_model.set_dq(ngroup=len(read_pattern), nborder=pars.nborder)

    #print(read_pattern)
    #print(len(read_pattern))
    #print('-->', counts.array[3890,237], e2dn_model.linearity_file, e2dn_model.gain_file, e2dn_model.ipc_file)
    #sys.stdout.flush()

    # generates resultants in DN
    resultants, dq = rstl1.apportion_counts_to_resultants(
        counts.array, tij, inv_linearity=e2dn_model, crparam={},
        persistence=persistence, tstart=tstart,
        rng=rng, seed=None)

    #print('resultants array', resultants[:,3890,237])

    with asdf.open(caldir['read']) as f:
        resultants = rstl1.add_read_noise_to_resultants(
            resultants, tij, rng=rng, seed=None,
            read_noise=f['roman']['data'][nb:-nb,nb:-nb],
            pedestal_extra_noise=None)

    if 'biascorr' in caldir:
        with asdf.open(caldir['biascorr']) as f:
            resultants += f['roman']['data']

    resultants[:,:,:] = np.round(resultants)

    return resultants*u.DN, dq

def noise_1f_frame(rng):
    """Generates a 4096x128 block of 1/f noise."""

    len = 2*pars.nside*pars.channelwidth

    this_array = np.zeros(2*len)
    galsim.GaussianDeviate(rng).generate(this_array)

    # get frequencies and amplitude ~ sqrt{power}
    freq = np.linspace(0,1-1./len,len)
    freq[len//2:] -= 1.
    amp = (1.e-99+np.abs(freq*len))**(-.5)
    amp[0] = 0.

    ftsignal = np.zeros((len,),dtype=np.complex128)
    ftsignal[:] = this_array[:len]
    ftsignal[:] += 1j*this_array[len:]
    ftsignal *= amp
    block = np.fft.fft(ftsignal).real[:len//2]/np.sqrt(2.)
    block -= np.mean(block)
    #print('generated noise, std -->', np.std(block))

    return(block.reshape((pars.nside,pars.channelwidth)).astype(np.float32))

def fill_in_refdata_and_1f(im, caldir, rng, tij, fill_in_banding=True, amp33=None):
    """Script to fill in the reference pixel data in an image.
    If amp33 is provided, then also tries to fill that in (provides a warning if not).

    im = the image data cube
    caldir = the calibration dictionary
    rng = random number generator
    tij = which reads to use
    nborder = number of reference pixels around the border
    fill_in_banding = also add 1/f noise?
    """

    (ngrp,ny,nx) = np.shape(im) # get shape
    nborder = parameters.nborder

    # the extra layer in noise is for the reset noise, which gets added to everything
    noise = np.zeros((ngrp+1,ny,nx), dtype=np.float32)
    galsim.GaussianDeviate(rng).generate(noise)
    with asdf.open(caldir['read']) as f:
        noise[:-1,:,:] *= f['roman']['data'][None,:,:]
        noise[-1,:,:] *= f['roman']['resetnoise']
    for j in range(len(tij)):
        noise[j,:,:] /= len(tij[j])**0.5
    noise[:-1,:,:] += noise[-1,:,:][None,:,:] # adds the reset noise to the reference pixels

    with asdf.open(caldir['dark']) as f:
        noise[:-1,:,:] += np.copy(f['roman']['data'])

    # what we have above is a dark image, but we want to fill in the
    # active pixels with the data from im
    noise[:-1,nborder:ny-nborder,nborder:nx-nborder] = im[:,nborder:ny-nborder,nborder:nx-nborder].astype(noise.dtype)

    # reference output
    amp33info = {
            'valid': False,
            'med': np.zeros((pars.nside,pars.channelwidth), dtype=np.float32),
            'std': np.zeros((pars.nside,pars.channelwidth), dtype=np.float32),
            'M_PINK': 0.,
            'RU_PINK': 0.
        }
    if amp33 is not None:
        with asdf.open(caldir['read']) as f:
            if 'amp33' in f['roman']:
                amp33info = copy.deepcopy(f['roman']['amp33'])
            else:
                warnings.warn("Didn't find reference output information. Skipping ...")

    # generate correlated noise
    if fill_in_banding:
        with asdf.open(caldir['read']) as f:
            u_pink = float(f['roman']['anc']['U_PINK'])
            c_pink = float(f['roman']['anc']['C_PINK'])
        print('adding correlated noise', u_pink, c_pink)
        for j in range(len(tij)):
            common_noise = noise_1f_frame(rng) * c_pink
            for ch in range(32):
                pinknoise = noise_1f_frame(rng) * u_pink + common_noise
                if ch%2==1: pinknoise = pinknoise[:,::-1]
                noise[j,:,pars.channelwidth*ch:pars.channelwidth*(ch+1)] += (pinknoise/len(tij[j])**0.5).astype(noise.dtype)

            # reference output is completely built here (signal + noise)
            if amp33info['valid']:
                whitenoise = np.zeros((pars.nside,pars.channelwidth),dtype=np.float32)
                galsim.GaussianDeviate(rng).generate(whitenoise)
                whitenoise *= amp33info['std']
                pinknoise = amp33info['RU_PINK']*noise_1f_frame(rng) + amp33info['M_PINK']*common_noise
                amp33[j,:,:] = (amp33info['med'] + (whitenoise+pinknoise)/len(tij[j])**0.5).astype(amp33.dtype)

    # write back to the original
    im[:,:,:] = np.clip(np.round(noise[:-1,:,:]),0,2**16-1).astype(im.dtype)

class Image2D:

    """This class has a 2D image, along with WCS and sky information.

    It can be constructed from simulations or (ultimately) from Roman data.

    Attributes:
    image : the 2D image
    galsimwcs : the world coordinate system for this image
    header : the world coordinate system for this image in FITS WCS format
    date : the observation date
    filter : the observation filter (4 characters, e.g., R062)
    idsca : ordered pair, (obs ID, SCA)
    ra_, dec_, pa_ : coordinates of the observation
    af, af2 : Level 1 & Level 2 files
    refdata : reference data

    Methods:
    __init__ : constructor
    init_anlsim : constructor from Open Universe simulation file.
    simulate : simulates the ramps, including L1 and L2 data.
    L1_write_to : write simulated L1 data file (ASDF)
    L2_write_to : write simulated L2 data file (ASDF)
    """

    def __init__(self, intype, **kwargs):
        """Wrapper for possible ways to build a 2D image.

        Possible input types:
        anlsim : from Open Universe 2024 simulation "truth" (or equivalent format)
        """

        if intype == 'anlsim':
             self.init_anlsim(kwargs['fname'])


    def init_anlsim(self, fname, flip=True):
        """If flip is True, then flips from SCA native coordinates to science-aligned
        (SOC-like product).
        """

        # get (id,sca)
        m = re.search(r'_(\d+)_(\d+)\.fits', fname)
        self.idsca = ( int(m.group(1)), int(m.group(2)) )

        # read header and data
        with fits.open(fname) as f:
            data = f[0].data
            self.header = f[0].header

        # flip SCAs depending on which row they are in
        if flip:
            if self.idsca[1]%3==0:
                hdu_sip_hflip(data,self.header)
            else:
                hdu_sip_vflip(data,self.header)

        self.image = data / self.header['EXPTIME'] # get this in electrons per second
        # offset from FITS -> GWCS convention
        self.header['CRPIX1'] -= 1; self.header['CRPIX2'] -= 1
        self.galsimwcs, origin = galsim.wcs.readFromFitsHeader(self.header)
        try:
            self.date = self.header['DATE-OBS']
            print(self.date)
            self.date = re.sub(' ', 'T', self.date)+'Z'
            print(self.date)
        except:
            self.date = '2025-01-01T00:00:00.000000'
        self.filter = self.header['FILTER'][:4]

        self.ra_ = float(self.header['RA_TARG'])
        self.dec_ = float(self.header['DEC_TARG'])
        self.pa_ = float(self.header['PA_OBSY'])

    def simulate(self, use_read_pattern, caldir=None, config={}, seed=43):
        """This is based on the romanisim.image.simulate function,
        but some functionality has been changed to be useful for this class.

        The "caldir" is a dictionary of where the calibration files are
        located.
        """

        target_pattern = 1000000
        parameters.read_pattern[target_pattern] = use_read_pattern
        metadata = ris.set_metadata(date = self.date, bandpass = self.filter,
           sca = self.idsca[1], ma_table_number=target_pattern)

        print('::',self.ra_,self.dec_,self.pa_)
        coord = SkyCoord(ra=self.ra_ * u.deg, dec=self.dec_ * u.deg,
                         frame='icrs')
        wcs.fill_in_parameters(metadata, coord, boresight=False, pa_aper=self.pa_)

        rng = galsim.UniformDeviate(seed)

        ### steps below are from romanisim.image.simulate ###

        meta = maker_utils.mk_common_meta()
        meta["photometry"] = maker_utils.mk_photometry()
        meta['wcs'] = None

        for key in parameters.default_parameters_dictionary.keys():
            meta[key].update(parameters.default_parameters_dictionary[key])

        for key in metadata.keys():
            meta[key].update(metadata[key])

        util.add_more_metadata(meta)

        # Create Image model to track validation
        image_node = maker_utils.mk_level2_image()
        image_node['meta'] = meta
        image_mod = roman_datamodels.datamodels.ImageModel(image_node)

        filter_name = image_mod.meta.instrument.optical_element

        read_pattern = metadata['exposure'].get('read_pattern', use_read_pattern)

        # for this simulation, we want to build something self-contained
        refdata = rimage.gather_reference_data(image_mod, usecrds=False)
        reffiles = refdata['reffiles']

        # persistence -> None
        persistence = rip.Persistence()

        # boder reference pixels
        nborder = parameters.nborder

        # simulate a blank image
        if caldir is None:
            counts, simcatobj = rimage.simulate_counts(
                image_mod.meta, [], rng=rng, usecrds=False, darkrate=refdata['dark'],
                stpsf=False, flat=refdata['flat'], psf_keywords=dict())
        else:
            # get dark current in DN/p/s
            with asdf.open(caldir['dark']) as f:
                this_dark = f['roman']['dark_slope'][nborder:-nborder,nborder:-nborder]
            # get flat field
            with asdf.open(caldir['flat']) as f:
                this_flat = f['roman']['data'][nborder:-nborder,nborder:-nborder]
            # convert to e/p/s
            with asdf.open(caldir['gain']) as f:
                this_dark = this_dark * f['roman']['data'][nborder:-nborder,nborder:-nborder]
                g = np.copy(f['roman']['data'][nborder:-nborder,nborder:-nborder]) # save gain for later
            # de-convolve the IPC kernel
            with asdf.open(caldir['ipc4d']) as f:
                this_dark = ipc_rev(this_dark, f['roman']['data'])           # dark is in e/s
                this_flat = ipc_rev(this_flat, f['roman']['data'], gain=g)   # but flat was measured in DN_lin so need gain=g
                this_flat = np.clip(this_flat,0.,2-2**-21)
                this_dark = np.clip(this_dark,-0.1*this_flat,None)           # prevent a spurious negative dark from giving an illegal negative total count rate
            # now run with this version of the dark rate
            counts, simcatobj = rimage.simulate_counts(
                image_mod.meta, [], rng=rng, usecrds=False, darkrate=this_dark,
                stpsf=False, flat=this_flat, psf_keywords=dict())
        util.update_pointing_and_wcsinfo_metadata(image_mod.meta, counts.wcs)

        # convert from e/s --> e using the parameters file and read pattern
        t = parameters.read_time * (use_read_pattern[-1][-1] - use_read_pattern[0][0])

        # the input simulations include the pixel area in their estimated e/s
        # but the flat field will ultimately include the pixel area as well!
        # therefore we need to re-scale the flat to the ideal pixel area (0.11 arcsec)^2
        flat_witharea = this_flat / ( pixelarea(self.galsimwcs,N=pars.nside_active)/pars.Omega_ideal )
        if 'CNORM' in config:
            C = float(config['CNORM'])
        else:
            C = 1.
        print(pixelarea(self.galsimwcs,N=pars.nside_active)[::1024,::1024])
        print(flat_witharea[::1024,::1024]); sys.stdout.flush()
        counts.array[:,:] += rng.np.poisson(lam=np.clip(C*t*g/pars.g_ideal*self.image*flat_witharea,0,None)).astype(counts.array.dtype)

        # this is where the (simulated) L1 data is created
        if caldir is None:
            l1, l1dq = rstl1.make_l1(
                counts, read_pattern, read_noise=refdata['readnoise'],
                pedestal_extra_noise=parameters.pedestal_extra_noise,
                rng=rng, gain=refdata['gain'],
                crparam={},
                inv_linearity=refdata['inverselinearity'],
                tstart=image_mod.meta.exposure.start_time,
                persistence=persistence,
                saturation=refdata['saturation'])
        else:
            # need to run the individual steps ourselves
            l1, l1dq = make_l1_fullcal(counts, read_pattern, caldir, rng=rng,
                tstart=image_mod.meta.exposure.start_time, persistence=persistence)

        # convert to asdf
        im, extras = rstl1.make_asdf(l1, dq=l1dq, metadata=image_mod.meta, persistence=persistence)

        # fill in the reference pixels and reference output
        if caldir is not None:

            # NO_AMP33 would allow us to bypass the reference output, if it isn't in the config file
            amp33struct = im['amp33']
            if 'NO_AMP33' in caldir:
                if caldir['NO_AMP33']:
                    amp33struct = None

            fill_in_refdata_and_1f(im['data'], caldir, rng, rstl1.read_pattern_to_tij(read_pattern), fill_in_banding=True, amp33=amp33struct)

        # get extras
        if reffiles:
            extras["simulate_reffiles"] = {}
            for key, value in reffiles.items():
                extras["simulate_reffiles"][key] = value
        extras['simcatobj'] = simcatobj
        extras['wcs'] = wcs.wcs_from_fits_header(self.header)
        # convert_wcs_to_gwcs(self.galsimwcs) # <-- a GWCS object! # wcs.convert_wcs_to_gwcs(counts.wcs)

        # Create metadata for simulation parameter
        romanisimdict = {'version': rstversion}
        romanisimdict.update(**extras)

        # Write file
        self.af = asdf.AsdfFile()
        self.af.tree = {'roman': im, 'romanisim': romanisimdict}

        # Make idealized L2 data
        slopeinfo = rimage.make_l2(l1, read_pattern, read_noise=refdata['readnoise'],
                            gain=refdata['gain'], flat=refdata['flat'], linearity=refdata['linearity'],
                            darkrate=refdata['dark'], dq=l1dq)
        l2dq = np.bitwise_or.reduce(l1dq, axis=0)

        # package header so that there is a obj.header.header
        # this is a hack for compatibility with convert_wcs_to_gwcs
        class Blank:
            pass
        obj = Blank()
        obj.header = Blank()
        obj.header.header = self.header
        #
        im2, extras2 = rimage.make_asdf(
            *slopeinfo, metadata=image_mod.meta, persistence=persistence,
            dq=l2dq, imwcs=obj, gain=refdata['gain'])
        # functionality to pull over the WCS from L1 without dependence on wcs.convert_wcs_to_gwcs
        #im2['roman']['meta'].update(wcs=this_gwcs)
        #im2['roman']['meta']['wcsinfo']['s_region'] = wcs.create_s_region(this_gwcs)

        # Create metadata for simulation parameter
        romanisimdict2 = {'version': rstversion}
        romanisimdict2.update(**extras2)

        # Write file
        self.af2 = asdf.AsdfFile()
        self.af2.tree = {'roman': im2, 'romanisim': romanisimdict2}

        self.refdata = refdata


    def L1_write_to(self, filename):
        """Writes to a file if there is an ASDF file present.
        Returns True if successful, False if not written.
        """
        if hasattr(self,'af'):
            self.af.write_to(open(filename, 'wb'))
        else:
            return False

    def L2_write_to(self, filename):
        """Writes to a file if there is an ASDF file present.
        Returns True if successful, False if not written.
        """
        if hasattr(self,'af2'):
            self.af2.write_to(open(filename, 'wb'))
        else:
            return False

class Image2D_from_L1(Image2D):
    """Similar to Image2D, but constructed from L1 data file.

    with Image2D_from_L1(infile, refdata, thewcs) as L1:
        ...
    """

    # Context manager functions
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.af.close()

    # Constructor
    def __init__(self, infile, refdata, thewcs, verbose_err=True):
        """Constructor. The arguments are:
        infile : L1 data file (ASDF format)
        refdata : the calibration reference data
        thewcs : WCS object (in some form -- currently FITS Header or GalSimWCS, though we've had issues with the latter)
        """

        self.af = asdf.open(infile)
        self.refdata = refdata
        self.thewcs = thewcs

    def psuedocalibrate(self):
        """Generates a simple calibrated (L2) image.

        This doesn't use romancal, but can be useful as a pass-through function.
        """

        # collect information
        nborder = parameters.nborder

        # Make idealized L2 data
        refdata = self.refdata
        l1dq = np.zeros(np.shape(self.af['roman']['data'][:,nborder:-nborder,nborder:-nborder]), dtype=np.uint32)
        slopeinfo = rimage.make_l2(self.af['roman']['data'][:,nborder:-nborder,nborder:-nborder]*u.DN,
                            self.af['roman']['meta']['exposure']['read_pattern'],
                            read_noise=refdata['readnoise'],
                            gain=refdata['gain'], flat=refdata['flat'], linearity=refdata['linearity'],
                            darkrate=refdata['dark'], dq=l1dq)
        l2dq = np.bitwise_or.reduce(l1dq, axis=0)

        # make WCS --- a few ways of doing this
        while True:
            wcsobj = None
            class Blank:
                pass

            # first try a FITS header
            if isinstance(self.thewcs, fits.Header):
                wcsobj = Blank()
                wcsobj.header = Blank()
                wcsobj.header.header = self.thewcs
                break

            # should work if this is a GalSim WCS
            try:
                header = fits.Header()
                self.thewcs.writeToFitsHeader(header, galsim.BoundsI(0,pars.nside_active,0,pars.nside_active))
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

        persistence = rip.Persistence()
        im2, extras2 = rimage.make_asdf(
            *slopeinfo, metadata=self.af['roman']['meta'], persistence=persistence,
            dq=l2dq, imwcs=wcsobj, gain=refdata['gain'])

        # Create metadata for simulation parameter
        romanisimdict2 = {'version': rstversion}
        romanisimdict2.update(**extras2)

        # Write file
        self.af2 = asdf.AsdfFile()
        self.af2.tree = {'roman': im2, 'romanisim': romanisimdict2}

def simpletest():
    """This is a simple script to convert Roman to L1/L2.
    For internal testing only, not production.
    """

    use_read_pattern = [[0], [1], [2,3], [4,5,6,7,8,9], [10,11,12,13,14,15], [16,17,18,19,20,21,22,23],
        [24,25,26,27,28,29,30,31,32,33], [34]]

    x = Image2D('anlsim', fname='/fs/scratch/PCON0003/cond0007/anl-run-in-prod/truth/Roman_WAS_truth_F184_14747_10.fits')
    print(x.galsimwcs)
    print(x.date, x.idsca)
    print('>>', x.image)
    x.simulate(use_read_pattern)
    x.L1_write_to('sim1.asdf')
    x.L2_write_to('sim2-direct.asdf')

    f = asdf.open('sim1.asdf')
    print(f.info())
    print('corners:')
    print(f['romanisim']['wcs'])
    print(f['romanisim']['wcs']((0,0,4087,4087),(0,4087,0,4087)))
    print(f['roman']['meta'])
    fits.PrimaryHDU(f['roman']['data']).writeto('L1.fits', overwrite=True)

    with Image2D_from_L1('sim1.asdf', x.refdata, x.header) as ff:
        ff.pseudocalibrate()
        ff.L2_write_to('sim2.asdf')

    f = asdf.open('sim2.asdf')
    print(f.info())
    print('corners:')
    print(f['roman']['meta']['wcs']((0,0,4087,4087),(0,4087,0,4087)))
    fits.PrimaryHDU(f['roman']['data']).writeto('L2.fits', overwrite=True)

def run_config(config):
    """This allows the L1 image construction to be called as a Python function instead of a stand-alone code."""

    # calibration files
    if 'CALDIR' in config:
        caldir = config['CALDIR']
    else:
        caldir = None

    print('Reading from <--', config['IN'])
    print('Writing to -->', config['OUT'])
    print('Using calibration data:')
    print(caldir)
    use_read_pattern = []
    ng = len(config['READS'])//2
    for j in range(ng):
        use_read_pattern.append(list(range(int(config['READS'][2*j]), int(config['READS'][2*j+1]))))
    print('Read pattern:', use_read_pattern)

    # Optional inputs
    seed=43
    if 'SEED' in config:
        seed=int(config['SEED'])

    x = Image2D('anlsim', fname=config['IN'])
    x.simulate(use_read_pattern, caldir=caldir, config=config, seed=seed)
    x.L1_write_to(config['OUT'])

    # header information for the WCS
    x.header['COMMENT'] = 'truth wcs from sim_to_isim'
    x.header.tofile(config['OUT'][:-5] + '_asdf_wcshead.txt', overwrite=True)

    # also write the FITS file for viewing
    if 'FITSOUT' in config:
        if config['FITSOUT']:
            image_out = np.zeros((ng,pars.nside,pars.nside_augmented), dtype=np.uint16)
            with asdf.open(config['OUT']) as f:
                image_out[:,:,:pars.nside] = f['roman']['data']
                image_out[:,:,pars.nside:] = f['roman']['amp33']
                fits.PrimaryHDU(image_out).writeto(config['OUT'][:-5] + '_asdf_to.fits', overwrite=True)

    # simpletest()

if __name__ == "__main__":
    """Stand-alone function to convert from OpenUniverse to L1. Call it with:

    python sim_to_isim <config file>

    The config file is in YAML format and has the fields:
    Required:
    'IN': input file name (FITS)
    'OUT': output file name (must end in '.asdf')
    'READS': a list of length 2*Ngrp: 0th group is [READS[0]:READS[1]], then [READS[2]:READS[3]], etc.

    Optional:
    'CALDIR': Python dictionary with the calibration reference files. This can contain:
        LINEARITY
    'FITSOUT': also write a FITS output (default: False; mostly useful for visualization in ds9)
    'SEED': RNG seed
    """

    # read settings
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    run_config(config)
