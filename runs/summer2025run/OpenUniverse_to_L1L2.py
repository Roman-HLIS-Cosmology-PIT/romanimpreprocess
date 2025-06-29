"""Driver to process lots of sims.
"""

import sys
import os
import io
from contextlib import redirect_stdout
from multiprocessing import Pool
import datetime

from ...from_sim import sim_to_isim
from ...L1_to_L2 import gen_cal_image

### --- Setup information ---
input_dir = '/fs/scratch/PCON0003/cond0007/anl-run-in-prod/truth/'
output_dir = '/fs/scratch/PCON0003/cond0007/summer2025/'
cal_dir = '/fs/scratch/PCON0003/cond0007/cal/'
tag = 'DUMMY20250521'
seed = 500 # starting seed
dseed = 10 # step for seed
Nmax = 4

use_sca = int(sys.argv[1])

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
        'FITSOUT': True
    }
    runlist.append((j,cfgs1,cfgs2))

    seed += dseed
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
    gen_cal_image.calibrateimage(c2)

for r in runlist:
    f(r)
    sys.stdout.flush()
