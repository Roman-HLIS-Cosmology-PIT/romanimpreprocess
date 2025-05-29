import sys
import numpy as np
import asdf
from datetime import datetime, timezone

"""This script makes a mask file."""

outfile = sys.argv[1]
sca = int(sys.argv[2])

dq = np.zeros((4096,4096), dtype=np.uint32)

# Reference pixels
dq[:4,:] |= 2**31
dq[-4:,:] |= 2**31
dq[:,:4] |= 2**31
dq[:,-4:] |= 2**31

# Low QE pixels
with asdf.open(outfile.replace('_mask_', '_linearitylegendre_')) as f:
    pflat = f['roman']['pflat'][0,:,:]

dq |= np.where(pflat<.5, 2**13, 0).astype(np.uint32)

# Hot pixels - right now 12.5 DN/s
# Warm pixels - right now 0.25 DN/s
with asdf.open(outfile.replace('_mask_', '_dark_')) as f:
    darkslope = f['roman']['dark_slope']
dq |= np.where(pflat>0.25, np.where(pflat>12.5, 2**11, 2**12), 0).astype(np.uint32)


tree = {'roman': {
    'meta': {
        'author': 'makemask.py',
        'description': 'makemask.py',
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
    'dq': dq
},
}

asdf.AsdfFile(tree).write_to(outfile)
