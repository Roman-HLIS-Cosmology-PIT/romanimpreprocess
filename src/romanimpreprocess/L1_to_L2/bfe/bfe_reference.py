'''
Functions to create BFE reference file for bfe_correction.py

TODO: implement everything.

'''
import numpy as np
from solid_waffle import ftsolve

records = []
raw_data = {}
grid_data = {}

COL_NUM = 26 

def extract_bfe_data(filename):
    with open(filename) as f:
        data = np.loadtxt(f)

        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        return data[:, COL_NUM]


def extract_summary_column(filename, col):
     with open(filename) as f:
             data = np.loadtxt(f)
     
             if data.ndim == 1:
                 data = data.reshape(1, -1)
             
             return data[:, col]

def deconv(filename, N):
    bfek = extract_summary_column(filename, 26)

    aD = extract_summary_column(filename, 11)
    aH = extract_summary_column(filename, 7)
    aV = extract_summary_column(filename, 8)

    aD_nl = extract_summary_column(filename, 11)
    aH_nl = extract_summary_column(filename, 7)
    aV_nl = extract_summary_column(filename, 8)
    
    k = ftsolve.decenter(ftsolve.pad_to_N(np.array([[aD, aV, aD], [aH, 1 - 4 * aD - 2 * aV - 2 * aH, aH], [aD, aV, aD]]), N))

    knl = ftsolve.decenter(
        ftsolve.pad_to_N(
            np.array(
                [
                    [aD_nl, aV_nl, aD_nl],
                    [aH_nl, -4 * aD_nl - 2 * aV_nl - 2 * aH_nl, aH_nl],
                    [aD_nl, aV_nl, aD_nl],
                ]
            ),
            N,
        )
    )

    # solve Fourier version for asq: F(BFEK) = Ksq^2*asq + Ksq*Knl_sq
    bfek = ftsolve.decenter(ftsolve.pad_to_N(bfek, N))
    ksq = ftsolve.fft2(k)
    knl_sq = ftsolve.fft2(knl)
    asq = (ftsolve.fft2(bfek) - ksq * knl_sq) / ksq**2
    return ftsolve.ifft2(asq)

def make_kernel(filename):
    a = deconv(filename, 5)
    np.reshape(a, (32,32))

