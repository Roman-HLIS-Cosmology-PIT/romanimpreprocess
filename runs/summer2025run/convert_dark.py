"""
Dark conversion script.

Calling format:

python convert_dark.py <input directory> <Number of frames> <output directory> <sca>

"""

import datetime
import re
import subprocess
import sys

import numpy
from astropy.io import fits

indir = sys.argv[1]
N = int(sys.argv[2])

sca = int(sys.argv[4])

for j in range(1, 101):
    procstring = indir + f"/Total_Noise_exp{j:d}_"
    print("Looking for files in", procstring)

    # get the files
    cube = numpy.zeros((1, N, 4096, 4224), dtype=numpy.uint16)
    date = []
    cmd = r"ls " + procstring + r"*" + f"SCU{sca:02d}" + r"*.fits"
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        files = p.stdout.decode("utf-8").split()
    except subprocess.CalledProcessError:
        print(f"no files SCU{sca:02d}, continuing ...")
        continue
    files.sort()
    #
    # strip out guide window files
    #
    files2 = []
    for f in files:
        if re.search(r"[0123456789ABCDEFabcdef]\.fits$", f):
            files2 += [f]
    files = files2
    #
    print(f"SCU{sca:02d}, found {len(files):3d} files")
    if len(files) < N:
        print(f"could not complete SCU{sca:02d}, continuing ...")
        continue

    # now let's open and read them
    for k in range(N):
        with fits.open(files[k]) as f:
            date += [f[0].header["DATE"]]
            cube[0, k, :, :] = f[0].data

    # raw data was in Detector frame, switch to science
    if True:
        if sca % 3 == 0:
            cube[:, :, :, :4096] = cube[:, :, :, 4095::-1]
        else:
            cube = cube[:, :, ::-1, :]

    # make an unweighted slope image, in DN/frame, dropping first frame
    slp = numpy.zeros((2, 4096, 4224))
    de = 0.0
    for k in range(1, N):
        slp[0, :, :] = slp[0, :, :] + cube[0, k, :, :] * (k - N / 2.0)
        de = de + (k - N / 2.0) ** 2
    slp[0, :, :] = slp[0, :, :] / de
    de = 0.0
    for k in range(1, N // 2):
        slp[1, :, :] = slp[1, :, :] + cube[0, k, :, :] * (k - (N // 2) / 2.0)
        de = de + (k - (N // 2) / 2.0) ** 2
    slp[1, :, :] = slp[1, :, :] / de

    # write to a FITS file
    outfile = sys.argv[3] + f"/99999999_SCA{sca:02d}_Noise_{j:03d}.fits"
    print(">>", outfile)
    hdr = fits.Header()
    hdr["PROVEN"] = ("convert_dark.py", "Conversion script")
    hdr["NMAX"] = (N, "max frames used")
    hdr["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for k in range(N):
        fname = files[k].split("/")[-1]
        print(fname)
        hdr[f"FR{k+1:03d}"] = fname
        hdr[f"FRD{k+1:03d}"] = (date[k], f"Timestamp, frame {k+1:d}")
    my_hdu = fits.ImageHDU(cube, header=hdr)
    hdr2 = fits.Header()
    hdr2["BUNIT"] = "DN/frame"
    my_hdu2 = fits.ImageHDU(slp.astype(numpy.float32), header=hdr2)
    prim = fits.PrimaryHDU()
    prim.header["TGROUP"] = 3.04
    fits.HDUList([prim, my_hdu, my_hdu2]).writeto(outfile, overwrite=True)
