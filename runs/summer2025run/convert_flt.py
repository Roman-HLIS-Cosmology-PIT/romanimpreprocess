"""Calling format:

python convert_flt.py <input directory> <Number of frames> <output directory> <sca>
"""

import datetime
import subprocess
import re
import sys
import numpy
from astropy.io import fits

indir = sys.argv[1]
N = int(sys.argv[2])

sca = int(sys.argv[4])

for j in range(1,51):
  procstring = indir + '/linearity_exp{:d}_'.format(j)
  print('Looking for files in', procstring)

  # get the files
  cube = numpy.zeros((1,N,4096,4224), dtype=numpy.uint16)
  date = []
  cmd = r'ls '+procstring+r'*'+'SCU{:02d}'.format(sca)+r'*.fits'
  try:
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = p.stdout.decode('utf-8').split()
  except:
    print('no files SCU{:02d}, continuing ...'.format(sca))
    continue
  files.sort()
  #
  # strip out guide window files
  #
  files2 = []
  for f in files:
    if re.search(r'[0123456789ABCDEFabcdef]\.fits$',f):
      files2 += [f]
  files = files2
  #
  print('SCU{:02d}, found {:3d} files'.format(sca,len(files)))
  if len(files)<N:
    print('could not complete SCU{:02d}, continuing ...'.format(sca))
    continue

  # now let's open and read them
  for k in range(N):
    with fits.open(files[k]) as f:
      date += [f[0].header['DATE']]
      cube[0,k,:,:] = f[0].data

  # raw data was in Detector frame, switch to science
  if True:
     if sca%3==0:
       cube = cube[:,:,:,::-1]
     else:
       cube = cube[:,:,::-1,:]

  # make an unweighted slope image, in DN/frame, dropping first frame
  Nc = N
  if N>10: Nc=10
  slp = numpy.zeros((2,4096,4224))
  de = 0.
  for k in range(1,Nc):
    slp[0,:,:] = slp[0,:,:] + cube[0,k,:,:]*(k-Nc/2.)
    de = de + (k-Nc/2.)**2
  slp[0,:,:] = slp[0,:,:]/de
  de = 0.
  for k in range(1,Nc//2):
    slp[1,:,:] = slp[1,:,:] + cube[0,k,:,:]*(k-(Nc//2)/2.)
    de = de + (k-(Nc//2)/2.)**2
  slp[1,:,:] = slp[1,:,:]/de

  # write to a FITS file
  outfile = sys.argv[3]+'/99999999_SCA{:02d}_Flat_{:03d}.fits'.format(sca, j)
  print('>>', outfile)
  hdr = fits.Header()
  hdr['PROVEN'] = ('convert_flt.py', 'Conversion script') 
  hdr['NMAX'] = (N, 'max frames used')
  hdr['DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  for k in range(N):
    fname = files[k].split('/')[-1]
    print(fname)
    hdr['FR{:03d}'.format(k+1)] = fname
    hdr['FRD{:03d}'.format(k+1)] = (date[k], 'Timestamp, frame {:d}'.format(k+1))
  my_hdu = fits.ImageHDU(cube, header=hdr)
  hdr2 = fits.Header()
  hdr2['BUNIT'] = 'DN/frame'
  my_hdu2 = fits.ImageHDU(slp.astype(numpy.float32), header=hdr2)
  prim = fits.PrimaryHDU()
  prim.header['TGROUP'] = 3.04
  fits.HDUList([prim, my_hdu, my_hdu2]).writeto(outfile, overwrite=True)
