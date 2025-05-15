"""Functions to convert external simulated images to Roman L1/L2-like format.

This works entirely at the single exposure level. Some parts wrap romanisim.
"""

import re
import numpy as np
import gwcs
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import asdf
import galsim
from galsim import roman

import roman_datamodels
import roman_datamodels.maker_utils as maker_utils
from romanisim import parameters, util, wcs, image as rimage, ris_make_utils as ris, persistence as rip, l1 as rstl1, __version__ as rstversion

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
        # the odd p's need a sign flip to reverse the direction of the SIP u-axis
        a_order = int(header['A_ORDER'])
        b_order = int(header['B_ORDER'])
        for p in range(1,a_order+1,2):
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
        # the odd p's need a sign flip to reverse the direction of the SIP u-axis
        a_order = int(header['A_ORDER'])
        b_order = int(header['B_ORDER'])
        for q in range(1,a_order+1,2):
           for p in range(a_order+1-q):
              keyword = 'A_{:1d}_{:1d}'.format(p,q)
              if keyword in header:
                  header[keyword] = -float(header[keyword])
        for q in range(1,b_order+1,2):
           for p in range(b_order+1-q):
              keyword = 'B_{:1d}_{:1d}'.format(p,q)
              if keyword in header:
                  header[keyword] = -float(header[keyword])
    except:
        print('Exception in SIP table, skipping ...')
        pass

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

    def simulate(self, seed=43):
        """This is based on the romanisim.image.simulate function,
        but some functionality has been changed to be useful for this class.

        Uses 2 consecutive seeds, so don't start this with, e.g., seed=20 and then seed=21!
        """

        use_read_pattern = [[0], [1], [2,3], [4,5,6,7,8,9], [10,11,12,13,14,15], [16,17,18,19,20,21,22,23],
            [24,25,26,27,28,29,30,31,32,33], [34]]
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

        # simulate a blank image
        counts, simcatobj = rimage.simulate_counts(
            image_mod.meta, [], rng=rng, usecrds=False, darkrate=refdata['dark'],
            stpsf=False, flat=refdata['flat'], psf_keywords=dict())
        util.update_pointing_and_wcsinfo_metadata(image_mod.meta, counts.wcs)

        # convert from e/s --> e using the parameters file and read pattern
        t = parameters.read_time * (use_read_pattern[-1][-1] - use_read_pattern[0][0])
        counts.array[:,:] += rng.np.poisson(lam=np.clip(t*self.image,0,None)).astype(counts.array.dtype)

        # this is where the (simulated) L1 data is created
        l1, l1dq = rstl1.make_l1(
            counts, read_pattern, read_noise=refdata['readnoise'],
            pedestal_extra_noise=parameters.pedestal_extra_noise,
            rng=rng, gain=refdata['gain'],
            crparam={},
            inv_linearity=refdata['inverselinearity'],
            tstart=image_mod.meta.exposure.start_time,
            persistence=persistence,
            saturation=refdata['saturation'])

        # convert to asdf
        im, extras = rstl1.make_asdf(l1, dq=l1dq, metadata=image_mod.meta, persistence=persistence)

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

if __name__ == "__main__":
    """This is a simple script to convert Roman to L1/L2."""

    x = Image2D('anlsim', fname='/fs/scratch/PCON0003/cond0007/anl-run-in-prod/truth/Roman_WAS_truth_F184_14747_10.fits')
    print(x.galsimwcs)
    print(x.date, x.idsca)
    print('>>', x.image)
    x.simulate()
    x.L1_write_to('sim1.asdf')
    x.L2_write_to('sim2.asdf')

    f = asdf.open('sim1.asdf')
    print(f.info())
    print('corners:')
    print(f['romanisim']['wcs'])
    print(f['romanisim']['wcs']((0,0,4087,4087),(0,4087,0,4087)))
    print(f['roman']['meta'])
    fits.PrimaryHDU(f['roman']['data']).writeto('L1.fits', overwrite=True)

    f = asdf.open('sim2.asdf')
    print(f.info())
    print('corners:')
    print(f['roman']['meta']['wcs']((0,0,4087,4087),(0,4087,0,4087)))
    fits.PrimaryHDU(f['roman']['data']).writeto('L2.fits', overwrite=True)
