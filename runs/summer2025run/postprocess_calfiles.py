import sys
import numpy as np
from astropy.io import fits
import asdf
from datetime import datetime, timezone
import json
import yaml

from loc_pars import g_ideal # get the ideal gain

"""This script makes a p-flat and saturation file from a linearitylegendre file."""

infile = sys.argv[1]
sca = int(sys.argv[2])
readpatternname = sys.argv[3]

gain = infile.replace('_linearitylegendre_', '_gain_')
outfile_flat = infile.replace('_linearitylegendre_', '_pflat_')

with asdf.open(infile) as f:
    with asdf.open(gain) as g:
        pflat = f['roman']['pflat'][0,:,:]
        pflat /= np.median(pflat) # normalize to 1

        """This scaling will be replaced in the future
        when we have a measurement of the L-flat.
        """
        pflat *= g_ideal/np.median(g['roman']['data'])

# clip and flag outliers -- usually low response
dq = np.zeros(np.shape(pflat), dtype=np.uint32)
dq |= np.where(pflat<.01, 1, 0).astype(np.uint32)
dq |= np.where(pflat>1.99, 1, 0).astype(np.uint32)
pflat = np.clip(pflat, .01, 1.99)

tree = {'roman': {
    'meta': {
        'author': 'postprocess_calfiles.py',
        'description': 'postprocess_calfiles.py',
        'instrument': {
            'detector': 'WFI{:02d}'.format(sca),
            'name': 'WFI'
        },
        'origin': 'PIT - romanimpreprocess',
        'date': datetime.now(timezone.utc).isoformat(),
        'pedigree': 'DUMMY',
        'reftype': 'PFLAT',
        'telescope': 'ROMAN',
        'useafter': '!time/time-1.2.0 2020-01-01T00:00:00.000'
    },
    'data': pflat.astype(np.float32),
    'dq': dq
},
'notes': {
    'src': infile
}
}

asdf.AsdfFile(tree).write_to(outfile_flat)

# tell the user about the key information
print('Pflat quality -->', np.count_nonzero(dq[4:-4,4:-4]), 'science pixels flagged.')
print('deciles -->', [np.round(np.percentile(pflat,10*i),5) for i in range(1,10)])

# saturation

outfile_sat = infile.replace('_linearitylegendre_', '_saturation_')

with asdf.open(infile) as f:
    Smax = np.clip(f['roman']['Smax'],1,65535).astype(np.float32)
    dq = np.where(f['roman']['Smax']>f['roman']['Sref'], 0, 1).astype(np.uint32)

tree = {'roman': {
    'meta': {
        'author': 'postprocess_calfiles.py',
        'description': 'postprocess_calfiles.py',
        'instrument': {
            'detector': 'WFI{:02d}'.format(sca),
            'name': 'WFI'
        },
        'origin': 'PIT - romanimpreprocess',
        'date': datetime.now(timezone.utc).isoformat(),
        'pedigree': 'DUMMY',
        'reftype': 'SATURATION',
        'telescope': 'ROMAN',
        'useafter': '!time/time-1.2.0 2020-01-01T00:00:00.000'
    },
    'data': Smax-1,
    'dq': dq
},
'notes': {
    'src': infile
}
}

# tell the user about saturation
print('saturation deciles -->', [np.percentile(tree['roman']['data'],10*i) for i in range(1,10)])

asdf.AsdfFile(tree).write_to(outfile_sat)

# now work on the bias correction.
# this is the dark image minus what you get by running the dark current forward in time.
# It could be handled in the simulation, but is easiest to do here to save time.

# get time per frame in seconds
with open('linearity_pars_{:02d}.json'.format(sca)) as f:
    lpars = json.load(f)
tframe = 3.04
if 'TFRAME' in lpars:
    tframe = float(lpars['TFRAME'])
bframe = 1
if 'BIAS' in lpars:
    if 'SLICE' in lpars['BIAS']:
        bframe = int(lpars['BIAS']['SLICE'])

# get read pattern
with open('settings_'+readpatternname+'.yaml') as f:
    readpattern = yaml.safe_load(f)['READS']
    ngrp = len(readpattern)//2

import loc_ipc_linearity as ipc_linearity
nb = 4
with asdf.open(infile.replace('_linearitylegendre_', '_dark_')) as fd:
    # get the predicted dark
    dark = fd['roman']['dark_slope'][nb:-nb,nb:-nb]*tframe # reference pixels trimmed, converted to DN/frame
    (ny,nx) = np.shape(dark)
    Sdark_predicted = np.zeros((ngrp, ny, nx), dtype=np.float32)
    xref = (int(readpattern[2*bframe])+int(readpattern[2*bframe+1])-1)/2.
    print('-----', xref)
    for j in range(ngrp):
        fr1 = int(readpattern[2*j])
        fr2 = int(readpattern[2*j+1])
        print('::', j, fr1, fr2)
        for x in range(fr1,fr2):
            signal,_ = ipc_linearity.invlinearity(dark*(x-xref), infile, origin=(nb,nb))
            Sdark_predicted[j,:,:] += signal
        Sdark_predicted[j,:,:] /= float(fr2-fr1)
    del dark

    # now get the true dark
    bias_corr = fd['roman']['data'][:,nb:-nb,nb:-nb] - Sdark_predicted

    print('-->')
    print(fd['roman']['data'][:,nb:-nb,nb:-nb][:,:4,:4])
    print('-->')
    print(Sdark_predicted[:,:4,:4])

# now save this
outfile_biascorr = infile.replace('_linearitylegendre_', '_biascorr_')
tree = {'roman': {
    'meta': {
        'author': 'postprocess_calfiles.py',
        'description': 'postprocess_calfiles.py',
        'instrument': {
            'detector': 'WFI{:02d}'.format(sca),
            'name': 'WFI'
        },
        'origin': 'PIT - romanimpreprocess',
        'date': datetime.now(timezone.utc).isoformat(),
        'pedigree': 'DUMMY',
        'reftype': 'BIASCORR',
        'telescope': 'ROMAN',
        'useafter': '!time/time-1.2.0 2020-01-01T00:00:00.000'
    },
    'data': bias_corr.astype(np.float32),
    't0': tframe*xref,
    't0_comment': 'number of seconds after reset used to define Sref, corresponding to 0 DN_lin'
}
}
asdf.AsdfFile(tree).write_to(outfile_biascorr)

B = np.zeros((2*ngrp,4088,4088), dtype=np.float32)
B[:ngrp,:,:] = bias_corr
B[ngrp:,:,:] = Sdark_predicted
fits.PrimaryHDU(B).writeto(outfile_biascorr[:-5]+'_asdf_to.fits', overwrite=True)
