"""Driver to process lots of sims.
"""

import sys
import os
import io
from contextlib import redirect_stdout
from multiprocessing import Pool
import datetime

from ...from_sim import sim_to_isim
from ...L1_to_L2 import gen_cal_image, gen_noise_image
from ...utils import maskhandling

### --- Setup information ---

# helper function to get commands of the form --keychar=value
def _getval(keychar, default=None):
    n = len(keychar)
    for a in sys.argv[1:]:
        if a[:n+3] == '--'+keychar+'=':
           return a[n+3:]
    return default

input_dir = _getval('in')
output_dir = _getval('out', default='.')
cal_dir = _getval('cal')
tag = _getval('tag')
seed = int(_getval('seed', default='500'))
dseed = int(_getval('dseed', default='10'))
temp_dir = os.getenv('TMPDIR', default=output_dir+'/L2')
use_sca = int(_getval('sca', default='1'))
Nmax = int(_getval('nmax', default='999')) # maximum number of chips to build

print('arguments:')
print('  input_dir =', input_dir)
print('  output_dir =', output_dir)
print('  cal_dir =', cal_dir)
print('  tag =', tag)
print('  seed =', seed)
print('  dseed =', dseed)
print('  temp_dir =', temp_dir)
print('  use_sca =', use_sca)
print('  Nmax =', Nmax)
sys.stdout.flush()

### --- now we have the information for this run ---

# space seeds for SCAs
nsca = 18
seed += dseed*use_sca

# make directories if SCA 1
if use_sca==1:
    try:
        os.mkdir(output_dir + '/L1')
    except:
        print('L1 directory already exists ...')
    try:
        os.mkdir(output_dir + '/L2')
    except:
        print('L2 directory already exists ...')

# useful function for finding calibration files
def findcal(ctype,sca):
    ctype_ = ctype
    if ctype=='flat': ctype_='pflat' # right now we don't have the L-flats
    return cal_dir + '/roman_wfi_' + ctype_ + '_' + tag + '_SCA{:02d}.asdf'.format(sca)

# figure out list of input files
runlist = []
outputs = []
j=0
for infile in os.listdir(input_dir):
    if infile[-5:].lower()!='.fits':
        continue

    # get (obsid,sca) from file name
    arr = infile.split('_')
    band = arr[-3]
    obsid = int(arr[-2])
    sca = int(arr[-1][:-5])

    if sca!=use_sca: continue

    # now we need to process this file
    print('\nProcessing: '+infile+'  obs={:d},sca={:d}'.format(obsid,sca))

    # Level 1 config
    caldir = {}
    ctypes = ['linearitylegendre', 'gain', 'dark', 'read', 'ipc4d', 'flat', 'biascorr']
    for ctype in ctypes:
        caldir[ctype] = findcal(ctype,sca)
    cfgs1 = {
        'IN': input_dir + '/' + infile,
        'OUT': output_dir + '/L1/sim_L1_{:s}_{:d}_{:d}.asdf'.format(band,obsid,sca),
        'READS': [0, 1, 1, 2, 2, 4, 4, 10, 10, 26, 26, 32, 32, 34, 34, 35],
        'FITSOUT': False,
        'CALDIR': caldir,
        'CNORM': 1.0,
        'SEED': seed
    }
    seed += dseed*nsca

    # Level 2 config
    caldir = {}
    ctypes = ['saturation', 'linearitylegendre', 'gain', 'dark', 'read', 'ipc4d', 'flat', 'biascorr', 'mask']
    for ctype in ctypes:
        caldir[ctype] = findcal(ctype,sca)
    cfgs2 = {
        'IN': output_dir + '/L1/sim_L1_{:s}_{:d}_{:d}.asdf'.format(band,obsid,sca),
        'OUT': output_dir + '/L2/sim_L2_{:s}_{:d}_{:d}.asdf'.format(band,obsid,sca),
        'FITSWCS': output_dir + '/L1/sim_L1_{:s}_{:d}_{:d}_asdf_wcshead.txt'.format(band,obsid,sca),
        'CALDIR': caldir,
        'RAMP_OPT_PARS': {'slope': 0.4, 'gain': 1.8, 'sigma_read': 7.},
        'JUMP_DETECT_PARS': {'SthreshA': 5.5, 'SthreshB': 4.5, 'IthreshA': 0.6, 'IthreshB': 600.},
        'FITSOUT': False,
        'NOISE': {
            'LAYER': ['RS2', 'RS2'],
            'TEMP': temp_dir + '/temp_{:s}_{:d}_{:d}.asdf'.format(band,obsid,sca),
            'SEED': seed,
            'OUT': output_dir + '/L2/sim_L2_{:s}_{:d}_{:d}_noise.asdf'.format(band,obsid,sca)
        }
    }
    runlist.append((j,cfgs1,cfgs2))

    seed += dseed*nsca
    j+=1

# now run the configurations

N = len(runlist)
if Nmax is not None:
    if N>Nmax:
        N=Nmax
        runlist = runlist[:N]
print(N, 'exposures')
sys.stdout.flush()

# simulation function (to be given to workers)
def f(r):
    (j,c1,c2) = r
    print(c1)
    sim_to_isim.run_config(c1)
    print(c2)
    gen_noise_image.calibrateimage(c2 | {'SLICEOUT': True})
    gen_noise_image.generate_all_noise(c2)
    print('write mask')
    maskhandling.PixelMask1.convert_file(c2['OUT'], c2['OUT'][:-5]+'_mask.fits')

for r in runlist:
    f(r)
    sys.stdout.flush()
