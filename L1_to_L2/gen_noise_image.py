import sys
import numpy as np
import asdf
import yaml
import galsim
import re
from copy import deepcopy
from astropy.io import fits

from .. import pars
from ..from_sim.sim_to_isim import fill_in_refdata_and_1f
from .gen_cal_image import calibrateimage
from ..utils import sky

def _get_subscript(arr, ch):
    """Helper function to take a character ch in the array, and return the string that
    goes out to but does not include the next capital letter. e.g.:
    _get_subscript('RS2Pg4', 'S') --> '2'
    _get_subscript('RS2Pg4', 'P') --> 'g4'
    """
    return re.split(r'(?=[A-Z])', arr.split(ch)[-1])[0]


def make_noise_cube(config, rng):
    """Makes alternative files with extra read noise, and then differences them to return
    a 'read noise only' slope image.

    Returns: numpy cube, size (N_noise,nside_active,nside_active)
    """

    # The number of noise realizations we need
    N_noise = len(config['NOISE']['LAYER'])
    noiseimage = np.zeros((N_noise, pars.nside_active, pars.nside_active), dtype=np.float32)

    # For each realization, we will load the L1 data file, add read noise, and write to file.
    for i_noise in range(N_noise):

        cmd = config['NOISE']['LAYER'][i_noise] # this noise layer command
        with asdf.open(config['IN']) as f_in:
            mytree = deepcopy(f_in.tree)        # load 'old' data from disk
        orig = np.copy(mytree['roman']['data']) # and make a copy that we won't modify
        nb = pars.nborder                       # shorthand for border width

        with asdf.open(config['OUT']) as f_orig:
            diff = np.zeros_like(f_orig['roman']['data'])

        # read noise simulated?
        if 'R' in cmd:
            noiseflags = _get_subscript(cmd,'R') # get information on what to simulate

            # if not adding, clear the input image
            if 'a' not in noiseflags:
                with asdf.open(config['CALDIR']['dark']) as fb:
                    mytree['roman']['data'] = np.copy(fb['roman']['data']).astype(mytree['roman']['data'].dtype)

                # write this to a file and calibrate it
                af = asdf.AsdfFile(mytree)
                af.write_to(open(config['NOISE']['TEMP'], 'wb'))
                del af
                config3 = deepcopy(config)
                config3['IN'] = config['NOISE']['TEMP']
                config3['OUT'] = config['NOISE']['TEMP'][:-5] + '_refL2.asdf'
                calibrateimage(config3)

            # white noise
            for k in range(len(mytree['roman']['meta']['exposure']['read_pattern'])):
                resultants = np.copy(mytree['roman']['data'][k,nb:-nb,nb:-nb].astype(np.float32))
                im = np.zeros_like(resultants)
                galsim.GaussianDeviate(rng).generate(im)
                with asdf.open(config['CALDIR']['read']) as fr:
                    im *= fr['roman']['data'][nb:-nb,nb:-nb] / np.sqrt(len(mytree['roman']['meta']['exposure']['read_pattern'][k]))
                resultants += im
                del im
                mytree['roman']['data'][k,nb:-nb,nb:-nb] = np.round(np.clip(resultants,0,2**16-1)).astype(mytree['roman']['data'].dtype)
                del resultants

            # correlated noise
            fill_in_refdata_and_1f(
                mytree['roman']['data'],                             # the data array
                config['CALDIR'],                                    # calibration data structure
                rng,                                                 # random number generator
                mytree['roman']['meta']['exposure']['read_pattern'], # readout scheme
                amp33=mytree['roman']['amp33']                       # reference output
            )

            # write to a temporary file
            af = asdf.AsdfFile(mytree)
            af.write_to(open(config['NOISE']['TEMP'], 'wb'))
            del af

            # now run L1-->L2 for this file
            config2 = deepcopy(config)
            config2['IN'] = config['NOISE']['TEMP']
            config2['OUT'] = config['NOISE']['TEMP'][:-5] + '_L2.asdf'
            calibrateimage(config2)

            # get difference
            with asdf.open(config2['OUT']) as f_out:
                origfile = config['OUT']
                if 'a' not in noiseflags: origfile = config3['OUT']
                with asdf.open(origfile) as f_orig:
                    diff = f_out['roman']['data'] - f_orig['roman']['data']

            # clip if requested
            if 'z' in noiseflags:
                zclip = float(_get_subscript(noiseflags.upper(),'Z'))
                IQR = np.percentile(diff,75)-np.percentile(diff,25)
                MED = np.percentile(diff,50)
                print('***', noiseflags, zclip, IQR, MED)
                diff = np.clip(diff, MED-zclip*IQR/1.34896, MED+zclip*IQR/1.34896)

        # Poisson noise simulated?
        if 'P' in cmd:
            noiseflags = _get_subscript(cmd,'P') # get information on what to simulate

            # first get the sky map. The 'b' flag chooses background only (with numerical order, if given).
            if 'b' in noiseflags:
                sky_order = int('0' + _get_subscript(noiseflags.upper(),'B'))
                with asdf.open(config['OUT']) as f_orig:
                    skylevel = sky.medfit(f_orig['roman']['data'], order=sky_order)[1]
            else:
                with asdf.open(config['OUT']) as f_orig:
                    skylevel = np.copy(f_orig['roman']['data'])

            # ramp-fitting weights
            ngrp = len(mytree['roman']['meta']['exposure']['read_pattern'])
            weightvecs = ['']*ngrp
            with asdf.open(config['OUT']) as f_L2:
                meta = f_L2['processinfo']['meta']
                weightvecs[-1] = np.copy(f_L2['processinfo']['weights'])
                start = 0
                if f_L2['processinfo']['exclude_first']: start=1
                for iend in range(start+2,ngrp):
                    Kt = np.zeros(ngrp,dtype=np.float32)
                    Kt[iend-1] = 1./(meta['tbar'][iend-1]-meta['tbar'][start])
                    Kt[start] = -Kt[iend-1]
                    weightvecs[iend-1] = Kt
                endslice = np.where(f_L2['processinfo']['endslice']>0, f_L2['processinfo']['endslice'], ngrp-1)
            print(weightvecs)

        # remove modes that would be taken out in sky subtraction
        if 'S' in cmd:
            sky_order = int('0' + _get_subscript(cmd,'S'))
            diff -= sky.medfit(diff, order=sky_order)[1]

        noiseimage[i_noise,:,:] = diff

    return noiseimage

def generate_all_noise(config):
    """Requires an additional 'NOISE' object in the configuration dictionary.
    It should have the entries:

    config['NOISE']['LAYER'] : list of noise realizations to build
    config['NOISE']['TEMP'] : temporary noise file location
    config['NOISE']['SEED'] : random number seed for the read noise images
    config['NOISE']['OUT'] : output of noise cube

    Layer commands start with a capital letter, then have lower case or numerical indications.

    R = include read noise
    S\d* = subtract sky using median filter of given order
    C... = reserved for comment (no capital letters in ...)

    The configuration has to have been run, since it looks for the L2 file written
    by gen_cal_image.calibrateimage.
    """

    rng = galsim.UniformDeviate(config['NOISE']['SEED'])
    noiseimage = make_noise_cube(config,rng)

    print(np.shape(noiseimage))
    print('percentiles:')
    for q in [5,25,50,75,95]:
        print(q, np.percentile(noiseimage,q,axis=(1,2)))

    # now output the noise image
    tree = {
        'config': config,
        'noise': noiseimage
    }
    af = asdf.AsdfFile(tree)
    af.write_to(open(config['NOISE']['OUT'], 'wb'))
    if 'FITSOUT' in config:
        if config['FITSOUT']:
            fitsout = fits.HDUList([fits.PrimaryHDU(noiseimage)])
            fitsout.writeto(config['NOISE']['OUT'][:-5]+'_asdf_to.fits', overwrite=True)

if __name__ == '__main__':
    """Stand-alone function processes L1->L2 and generates noise"""

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    calibrateimage(config | {'SLICEOUT':True}) # add slice information
    generate_all_noise(config)
