# Module imports
import numpy as np
import sys
import yaml
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, pearson3, t


# TO DO:

# 1. Generate distribution from tilde{nu}_pl's DONE

# 2. Confirm tilde{nu}_41 and tilde{nu}_21 relationship and improve efficiency DONE?
# 3. Compare efficiency to Monte Carlo approach
# 4. (Less important) Improve efficiency by removing zeroes (sparse matrices scipy.sparse?)



def raw_weights(N_beta,a_beta):
    '''
    Generates the matrix L to transform raw frames into MA frames.

    Used in this code to calculate the "compressed" moments of the MA frames
    using the moments of the raw frames.

    Inputs:
    N_beta -- a 1D numpy vector describing how many frames we will sum over to make an MA frame
    a_beta -- a 1D numpy vector indicating which frame index we start our sums

    Output:
    (L) raw weights of size (M,N)
    '''
    assert len(N_beta) == len(a_beta) # N_beta and a_beta must have same length

    N = np.max(a_beta + N_beta) # Number of raw frames, including skipped frames, starting from frame zero
    
    M = len(N_beta) # Number of MA frames

    L = np.zeros((M, N))

    for k in range(M):
        a_start = a_beta[k]
        a_end = a_start + N_beta[k]
        L[k, a_start:a_end] = 1.0 / N_beta[k]
    
    return L






def get_tilde_nus(N_beta,a_beta,W):
    '''
    Calculates the tilde{nu}_pl's used in computing the
    2nd, 3rd, and 4th moments of our xi distribution.

    Inputs:
    N_beta -- a 1D numpy vector describing how many frames we will sum over to make an MA frame
    a_beta -- a 1D numpy vector detailing which frame index we start our sums
    '''
    
    L = raw_weights(N_beta,a_beta)

    T = np.cumsum(L[:,::-1],axis=1)[:,::-1] # Uses raw weights to generate the central moments of raw frames (ignoring I)

    WT = np.dot(W,T[:,1:])                  # Uses input weights to help generate the

    nu_21 = np.sum(WT**2)                   # Uses the central moments of the MA frames to find the central moments of our final frames
    nu_31 = np.sum(WT**3)
    nu_41 = np.sum(WT**4)
    nu_42 = 3 * nu_21**2
    
    # Taken from equation (currently) 32 in "Noise bias and debiasing in photometric surveys"

    tilnu_21 = nu_21
    tilnu_31 = nu_31 - 3*nu_21**2
    tilnu_41 = nu_41 - 10*nu_21*nu_31 - nu_21*nu_42 + 18*nu_21**3
    tilnu_42 = nu_42
    
    return tilnu_21, tilnu_31, tilnu_41, tilnu_42



#EXAMPLE INPUTS:
'''
N_beta = np.array([4,4])
a_beta = np.array([5,12])
weights = np.array([-0.5, 0.5])

EXPLANATION:
We have 16 raw frames. Frames 5-8 and 12-15 are being compressed into 2 "MA" frames,
each with equal weights of -0.5 and 0.5. We disregard the other raw frames.
'''

# From Laliotis et al. 2024 Analysis of biasing from noise from the Nancy Grace Roman Space Telescope
N_beta = np.array([1,2,4,4,4,1])
a_beta = np.array([2,3,5,23,44,49])
weights = np.array([-0.1,-0.4,-0.2,0.2,0.4,0.1]) # Arbitrary weights

'''
N_beta = np.array([4,4,4,1])
a_beta = np.array([5,23,44,49])
weights = np.array([-0.2,-0.4,0.4,0.2])'''

tilnu_21, tilnu_31, tilnu_41, tilnu_42 = get_tilde_nus(N_beta,a_beta,weights)

print(f"𝜈̃₂₁ := {tilnu_21}")
print(f"𝜈̃₃₁ := {tilnu_31}")
print(f"𝜈̃₄₁ := {tilnu_41}")
print(f"𝜈̃₄₂ := {tilnu_42}")

# Seems like til{nu}_42 = 3 x til{nu}_21^2