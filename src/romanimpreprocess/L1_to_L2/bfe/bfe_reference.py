'''
Functions to create BFE reference file for bfe_correction.py

TODO: implement everything.

'''
import numpy as np
from solid_waffle import ftsolve

def extract_summary_column(filename, col):
     with open(filename) as f:
             data = np.loadtxt(f)
     
             if data.ndim == 1:
                 data = data.reshape(1, -1)
             
             return data[:, col]

def deconv(filename, N, bfek):

    aD = extract_summary_column(filename, 11)
    aH = extract_summary_column(filename, 7)
    aV = extract_summary_column(filename, 8)

    a = []
    for i in range(len(bfek)):
        k = ftsolve.decenter(ftsolve.pad_to_N(np.array([[aD[i], aV[i], aD[i]], [aH[i], 1 - 4 * aD[i] - 2 * aV[i] - 2 * aH[i], aH[i]], [aD[i], aV[i], aD[i]]]), N))

        # solve Fourier version for asq: F(BFEK) = Ksq^2*asq + Ksq*Knl_sq
        b = ftsolve.decenter(ftsolve.pad_to_N(bfek[i], N))
        ksq = ftsolve.fft2(k)
        asq = (ftsolve.fft2(b)) / ksq**2
        a.append(ftsolve.ifft2(asq))
    return np.array(a)

def make_kernel(filename):
    columns = [extract_summary_column(filename, col) for col in range(14, 39)]

    data = np.array(columns)
    data = data.T
    bfek = data.reshape(1024, 5, 5) 

    print(np.shape(bfek))
    print(np.shape(bfek))
    a = deconv(filename, 5, bfek)
    kernel = np.nanmedian(a, axis=0)
    print(kernel)
    print(np.shape(kernel))
    return kernel

make_kernel("src/romanimpreprocess/L1_to_L2/bfe/icemon-car216-pass85-wfi06_summary.txt")

