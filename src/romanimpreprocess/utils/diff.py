import sys

import asdf
import numpy as np
from astropy.io import fits

if len(sys.argv) < 5:
    print("Calling format: python diff.py <asdf in> <fits out> <group1> <group2>")
    exit()

with asdf.open(sys.argv[1]) as f:
    data = f["roman"]["data"].astype(np.float32)

diffimage = data[int(sys.argv[4]), :, :] - data[int(sys.argv[3]), :, :]

fits.PrimaryHDU(diffimage).writeto(sys.argv[2], overwrite=True)
