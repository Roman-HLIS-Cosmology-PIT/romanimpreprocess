"""Tools for building de-noising realizations from a science image.

This module was written by M. Gabe.
A few wrapper functions added by C. Hirata.
"""

# Module imports

import numpy as np

# import yaml
# import matplotlib.pyplot as plt


def raw_weights(N_beta, a_beta):
    """
    Generates the matrix L to transform raw frames into MA frames.

    Used in this code to calculate the "compressed" moments of the MA frames
    using the moments of the raw frames.

    Inputs:
    N_beta -- a 1D numpy vector describing how many frames we will sum over to make an MA frame
    a_beta -- a 1D numpy vector detailing which frame index we start our sums

    Output:
    (L) raw weights of size (M,N)
    """
    assert len(N_beta) == len(a_beta)  # N_beta and a_beta must have same length

    N = np.max(a_beta + N_beta)  # Number of raw frames, including skipped frames, starting from frame zero

    M = len(N_beta)  # Number of MA frames

    L = np.zeros((M, N))

    for k in range(M):
        a_start = a_beta[k]
        a_end = a_start + N_beta[k]
        L[k, a_start:a_end] = 1.0 / N_beta[k]

    return L


def centmoms(N_beta, a_beta):
    """
    Calculates the central moments of our raw frames.

    This function does not include Ibar

    For example,
    2nd central moment = np.min(a,b) Ibar

    but in this code,
    2nd central moment = np.min(a,b)

    as our nu_pl's are independent of the value of Ibar

    Inputs:
    N_beta -- a 1D numpy vector describing how many frames we will sum over to make an MA frame
    a_beta -- a 1D numpy vector detailing which frame index we start our sums

    Outputs:
    (centmom2) 2nd central moment / Ibar
    (centmom3) 3rd central moment / Ibar
    (centmom4lin) 4th central moment term linear in Ibar / Ibar
    (centmom4quad) 4th central moment quadratic in Ibar / Ibar^2
    """
    N = np.max(N_beta + a_beta)

    frame_inds = np.arange(0, N)

    i2, j2 = np.meshgrid(frame_inds, frame_inds)  # Two meshgrid indices for 2d
    centmom2 = np.minimum(i2, j2)  # Shape (N,N)

    i3, j3, k3 = np.meshgrid(
        frame_inds, frame_inds, frame_inds, indexing="ij"
    )  # Three meshgrid indices for 3d
    centmom3 = np.minimum.reduce([i3, j3, k3])  # Shape (N,N,N)

    i4, j4, k4, l4 = np.meshgrid(
        frame_inds, frame_inds, frame_inds, frame_inds, indexing="ij"
    )  # Four meshgrid indices for 4d

    centmom4lin = np.minimum.reduce([i4, j4, k4, l4])  # Shape (N,N,N,N)
    centmom4quad = (
        np.minimum(i4, j4) * np.minimum(k4, l4)
        + np.minimum(i4, k4) * np.minimum(j4, l4)
        + np.minimum(i4, l4) * np.minimum(j4, k4)
    )  # Shape (N,N,N,N)

    return centmom2, centmom3, centmom4lin, centmom4quad


def compress_moms(mom2, mom3, mom4lin, mom4quad, L):
    """
    Linearly compresses raw moments into MA moments using the raw weights L.

    Used in this script to find the central moments of our MA frames
    using the central moments of our raw frames and their raw weights, L.

    In this code, we treat the nu_pl's as equal to the
    compressed moments of our MA frames, as they differ by
    a factor of Ibar or Ibar^2, which cancel out when
    being compressed.

    Inputs:
    (centmom2) 2nd central moment / Ibar
    (centmom3) 3rd central moment / Ibar
    (centmom4lin) 4th central moment term linear in Ibar / Ibar
    (centmom4quad) 4th central moment quadratic in Ibar / Ibar^2

    Outputs:
    mom2x = compressed centmom2
    mom3x = compressed centmom3
    mom4linx = compressed centmom4lin
    mom4quadx = compressed centmom4quad

    """
    mom2x = np.einsum("ia,jb,ab->ij", L, L, mom2)  # shape (M,M)
    mom3x = np.einsum("ia,jb,kc,abc->ijk", L, L, L, mom3)  # shape (M,M,M)
    mom4linx = np.einsum("ia,jb,kc,ld,abcd->ijkl", L, L, L, L, mom4lin)  # shape (M,M,M,M)
    mom4quadx = np.einsum("ia,jb,kc,ld,abcd->ijkl", L, L, L, L, mom4quad)  # shape (M,M,M,M)

    return mom2x, mom3x, mom4linx, mom4quadx


def get_nus(N_beta, a_beta):
    """
    Finds nu_pl to 4th order

    Inputs:
    N_beta -- a 1D numpy vector describing how many frames we will sum over to make an MA frame
    a_beta -- a 1D numpy vector detailing which frame index we start our sums
    """

    # Obtain the weights required to turn raw frames
    # into MA frames
    L = raw_weights(N_beta, a_beta)

    # Obtain the central moments our raw frames
    centmom2, centmom3, centmom4lin, centmom4quad = centmoms(N_beta, a_beta)

    # Obtain the central moments of our MA frames by compressing
    # the central moments of our raw frames.
    nu_21, nu_31, nu_41, nu_42 = compress_moms(centmom2, centmom3, centmom4lin, centmom4quad, L)

    return nu_21, nu_31, nu_41, nu_42


def compress_nu(nu, W):
    """
    Compresses a nu_pl tensor of rank 2–4 using MA weights, W.

    Inputs:
    nu_pl (hypercubic array of dimension p with side length N)
    (W) MA weights (size N)

    Outputs scalars based on the dimension of nu_pl:
    W^a W^b         [nu_2l]_a,b
    W^a W^b W^c     [nu_3l]_a,b,c
    W^a W^b W^c W^d [nu_4l]_a,b,c,d
    """

    assert np.isclose(np.sum(W), 0.0, atol=1e-12)  # Weights must sum to zero

    if nu.ndim == 2:
        return np.einsum("a,b,ab->", W, W, nu)
    elif nu.ndim == 3:
        return np.einsum("a,b,c,abc->", W, W, W, nu)
    elif nu.ndim == 4:
        return np.einsum("a,b,c,d,abcd->", W, W, W, W, nu)


def get_tilde_nus(N_beta, a_beta, W):
    """
    Calculates the tilde{nu}_pl's used in computing the
    2nd, 3rd, and 4th moments of our xi distribution.

    Inputs:
    N_beta -- a 1D numpy vector describing how many frames we will sum over to make an MA frame
    a_beta -- a 1D numpy vector detailing which frame index we start our sums
    """

    # Use the get_nus function to get our nus
    nu_21, nu_31, nu_41, nu_42 = get_nus(N_beta, a_beta)

    # Use the compress_nu function
    # to turn our nus into scalars
    nu_21x = compress_nu(nu_21, W)
    nu_31x = compress_nu(nu_31, W)
    nu_41x = compress_nu(nu_41, W)
    nu_42x = compress_nu(nu_42, W)

    # Taken from equation (currently) 32 in "Noise bias and debiasing in photometric surveys"
    tilnu_21 = nu_21x
    tilnu_31 = nu_31x - 3 * nu_21x**2
    tilnu_41 = nu_41x - 10 * nu_21x * nu_31x - nu_21x * nu_42x + 18 * nu_21x**3
    tilnu_42 = nu_42x

    return tilnu_21, tilnu_31, tilnu_41, tilnu_42


def get_tilde_nus_from_list(grps, wt):
    """
    Similar, but with input of a grp list instead. Requires groups to be consecutive lists
    (drops between groups OK, drops within group not OK).
    """

    ngrp = len(grps)
    N_beta = np.zeros((ngrp,), dtype=np.int32)
    a_beta = np.zeros((ngrp,), dtype=np.int32)
    for i in range(ngrp):
        a_beta[i] = grps[i][0]
        N_beta[i] = len(grps[i])
    return get_tilde_nus(N_beta, a_beta, wt)


weights = np.array(
    [
        0.0000000e00,
        -2.1521233e-03,
        -3.6145949e-03,
        -6.7949751e-03,
        3.0364664e-10,
        6.7949742e-03,
        3.6145954e-03,
        2.1521233e-03,
    ]
)
weights[-1] = -np.sum(weights[:-1])
tilnu_21, tilnu_31, tilnu_41, tilnu_42 = get_tilde_nus_from_list(
    [
        [0],
        [1],
        [2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30, 31],
        [32, 33],
        [34],
    ],
    weights,
)

print(f"tnu21 := {tilnu_21}")
print(f"tnu31 := {tilnu_31}")
print(f"tnu41 := {tilnu_41}")
print(f"tnu42:= {tilnu_42}")
