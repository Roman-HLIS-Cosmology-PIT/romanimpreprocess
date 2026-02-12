'''

Introducing the function moms_from_Pearson:

Conceptually:
1. Input tilnu_pl's and intensities (identical to draw_from_tilnus).
2. Identifies which Pearson distribution families correspond to the tilnu_pl's and intensities
(identical to draw_from_tilnus).

3. Calculates corresponding parameters in the Pearson distribition corresponding to the tilnu_pl's and intensities
(conceptually identical to draw_from_tilnus -- for Pearson types 1, 5, and 6, we import
the function draw_from_tilnus uses for calculation. For types 3 and 4, we create different
functions that are identical in calculation.)

4. Uses the parameters calculated in step 3 to analytically find the moments if we had created a distribution (doesn't happen in draw_from_tilnus).

===========================================================================================================

1. moms_from_Pearson finds the analytical moments if we had created a distribution.
2. make_analymoms_mat iterates over moms_from_Pearson to create a matrix of moments from multiple inputs of tilnu_pl's and intensities.
2. target_moms finds the target moments and makes a matrix
3. make_err_mat compares the outputs of target_moms and moms_from_Pearson to calculate the relative error between the two


These functions can be used to confirm if the Pearson parameters calculated in draw_from_tilnus.py is self-consistent.

Can generally be used to identify the limitations of the Pearson distribution family.

Outputs text files containing the relative error between the required moments and the calculated moments.
Text files typically (hopefully) contains rows and columns of 0.0 and 1.0.

Regions of 1.0 can be identified by inspection and the user can identify the corresponding tilnu_pl's.
Regions of 1.0 should (hopefully) correspond to implausible distributions. See example at end of file.

'''


import numpy as np

from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import solve_beta_parameters_vec
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import solve_pearson5_parameters_vec
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import solve_pearson6_params


# ============================================================
# ============================================================

# Helper functions with analytical solutions of central moments

# ============================================================
# ============================================================

def analymoms_type1(tilnu_21,tilnu_31,tilnu_41,I_arr):
    '''
    Returns the analytical second, third, and fourth moments
    of the Beta distribution, given the two shape
    parameters alpha and beta.
    '''

    I_arr   = np.asarray(I_arr, dtype=float)
    tilnu_42 = 3.0 * tilnu_21 ** 2


    a, b, _, scale = solve_beta_parameters_vec(tilnu_21, tilnu_31, tilnu_41, I_arr)

    skew = 2 * (b-a) * np.sqrt(a+b+1) / ((a+b+2)*np.sqrt(a*b))
    kurt = (6* ( (a-b)**2 * (a + b + 1) - a*b*(a+b+2) ) / (a*b * (a+b+2) * (a+b+3) ) + 3)

    mu2 = (a * b) / ( (a + b)**2 * (a + b + 1) )
    mu3 = skew * mu2**(3/2)
    mu4 = kurt * mu2**2



    mu2s = mu2 * scale**2
    mu3s = mu3 * scale**3
    mu4s = mu4 * scale**4

    return mu2s, mu3s, mu4s

def analymoms_type3(tilnu_21: float,
    tilnu_31: float,
    I_vals: np.ndarray,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    I_vals = np.asarray(I_vals, dtype=float)

    scale  = abs(tilnu_31) / (2.0 * tilnu_21)                 # common scale
    shapes = 4.0 * tilnu_21**3 * I_vals / tilnu_31**2         # Γ shape per-pixel

    skew = 2/np.sqrt(shapes)
    kurt = 6/shapes + 3
    sign   = 1 if tilnu_31 > 0 else -1

    mu2 = shapes*scale**2
    mu3 = skew * mu2**(3/2)
    mu4 = kurt * mu2**2

    return mu2, sign*mu3, mu4

def analymoms_type4(tilnu21: float,
    tilnu31: float,
    tilnu41: float,
    I_vals: np.ndarray,
    rng=None,
):
    

    I_arr    = np.asarray(I_vals, dtype=float)
    flat_I   = I_arr.ravel()

    # --- vectorised parameter computation ---
    tilnu42 = 3*tilnu21**2
    mu2     = tilnu21 * flat_I
    beta1   = tilnu31**2 / (tilnu21**3 * flat_I)
    beta2   = (tilnu42*flat_I + tilnu41) / (tilnu21**2 * flat_I)

    denom = 2*beta2 - 3*beta1 - 6
    r     = 6*(beta2 - beta1 - 1) / denom
    inner = 16*(r-1) - beta1*(r-2)**2 # Whatever's inside the sqrt

    if np.any(r <= 3) or np.any(inner <= 0):
        raise ValueError("Some intensities give invalid Pearson-IV parameters.")

    nu_mag = r*(r-2)*np.sqrt(beta1) / np.sqrt(inner)
    sign   = -1.0 if tilnu31 >= 0 else 1.0   # sign(µ3)=−sign(ν)
    nu     = sign * nu_mag

    a      = np.sqrt(mu2*inner)/4

    mu2 = a**2 * (r**2 + nu**2) / ( (r**2)*(r-1) )
    mu3 = - 4 * a**3 * nu * (r**2 + nu**2) / ( r**3 * (r-1) * (r-2) )
    mu4 = 3*a**4 * (r**2 + nu**2) * ( (r+6) * (r**2 + nu**2) - 8*r**2 ) / (r**4 * (r-1)*(r-2)*(r-3))

    shape = I_arr.shape

    """print('nu =', nu)
    print(' r = ', r)

    print('a = ', a)
    print('')"""

    return mu2.reshape(shape), mu3.reshape(shape), mu4.reshape(shape)

def analymoms_type5(tilnu21: float,
    tilnu31: float,
    I_vals: np.ndarray,
    rng=None,
):

    a, b, _ = solve_pearson5_parameters_vec(
        tilnu21, tilnu31, I_vals
    )

    skew = 4*np.sqrt(a-2) / (a-3)
    kurt = 6 * (5*a-11) / ((a-3)*(a-4)) + 3

    mu2 = b**2 / ((a-1)**2 * (a-2))
    mu3 = skew * mu2**(3/2)
    mu4 = kurt * mu2**2
    return mu2, mu3, mu4

def analymoms_type6(tilnu21: float,
    tilnu31: float,
    tilnu41: float,
    I_vals: np.ndarray,
    rng=None,
):
    

    a, b, scale, shift, sgn = solve_pearson6_params(tilnu21, tilnu31, tilnu41, I_vals)
    
    skew = 2 * (2*a+b-1) / (b-3) * np.sqrt((b-2) / (a*(a+b-1)))
    ex_kurt_num = 6*( a*(a+b-1)*(5*b-11) + (b-1)**2 * (b-2) )
    ex_kurt_denom = a*(a+b-1)*(b-3)*(b-4) 
    kurt = ex_kurt_num/ex_kurt_denom + 3

    mu2 = a * (a+b-1) / ((b-2)*(b-1)**2)
    mu3 = skew * mu2**(3/2)
    mu4 = kurt * mu2**2

    mu2s = mu2 * scale**2
    mu3s = sgn*mu3 * scale**3
    mu4s = mu4 * scale**4

    return mu2s, mu3s, mu4s



# ============================================================
# ============================================================

# Functions for analytical testing of moments

# ============================================================
# ============================================================


def moms_from_Pearson(tilnu_21: float, tilnu_31: float, tilnu_41: float,
    I: np.ndarray, *, atol: float = 1e-14):
    """
    Compute analytic central moments (μ2, μ3, μ4) per element of 1D I
    using Pearson-type regions determined by (β1, β2).

    Parameters
    ----------
    tilnu_21, tilnu_31, tilnu_41 : float
        Scalar central-moment ratios (𝜈̃₂₁, 𝜈̃₃₁, 𝜈̃₄₁).
    I : ndarray
        Input intensities.
    atol : float, optional
        Equality tolerance for the region boundaries.

    Returns
    -------
    analymoms: Analytical moments shape(len(I),3)
    """

    I_clipped = np.clip(np.asarray(I, dtype=float), 3.0, None)

    

    tilnu_42 = 3 * tilnu_21 ** 2

    # Pearson variables
    b_1 = tilnu_31 ** 2 / (tilnu_21 ** 3 * I_clipped)                       # β₁
    b_2 = (tilnu_42 * I_clipped + tilnu_41) / (tilnu_21 ** 2 * I_clipped)   # β₂

    # ---  Pearson admissability  ----------------------------------

    # Base admissibility region (elements outside *base* stay unchanged.)
    base = (b_2 > 0) & (b_1 >= 0) & (b_2 > b_1 + 1) & (b_2 > 0.75 * b_1)

    # Handy aliases for the region lines
    lhs  = b_2
    rhs1 = 1.5 * b_1 + 3
    rhs2 = (48 + 39 * b_1 + 6 * (4 + b_1) ** 1.5) / (32 - b_1)
    #rhs3 = 1.875 * b_1 + 4.5      # 15/8 b₁ + 36/8

    # Equality bands (exclusive)
    eq1 = np.isclose(lhs, rhs1, atol=atol, rtol=0)        # → Type 3
    eq2 = np.isclose(lhs, rhs2, atol=atol, rtol=0)        # → Type 5

    # Strict zones shrunk by ±atol
    lt1 = lhs <  rhs1 - atol
    gt1 = lhs >  rhs1 + atol
    lt2 = lhs <  rhs2 - atol
    gt2 = lhs >  rhs2 + atol
    #lt3 = lhs <  rhs3 - atol

    # Masks: mutually exclusive
    type1 = base & lt1
    type3 = base & eq1
    type5 = base & eq2
    type6 = base & gt1 & lt2
    type4 = base & gt2# & lt3


    # TEST
    overlap = (
    type1.astype(int) + type3.astype(int) + type5.astype(int) +
    type6.astype(int) + type4.astype(int)
    ) > 1
    if np.any(overlap):
        print('ERROR:')
        print('tilnu_21: ',tilnu_21)
        print('tilnu_31: ',tilnu_31)
        print('tilnu_41: ',tilnu_41)
        raise RuntimeError("Overlapping Pearson masks.")

    # ---  Draws and in-place addition  ----------------------------------
    analymoms = np.zeros((len(I_clipped), 3), dtype=float)

    if np.any(type1):
        m2, m3, m4 = analymoms_type1(tilnu_21, tilnu_31, tilnu_41, I_clipped[type1])
        analymoms[type1,0] = m2; analymoms[type1,1] = m3; analymoms[type1,2] = m4
    if np.any(type3):
        m2, m3, m4 = analymoms_type3(tilnu_21, tilnu_31          , I_clipped[type3])
        analymoms[type3,0] = m2; analymoms[type3,1] = m3; analymoms[type3,2] = m4
    if np.any(type4):
        m2, m3, m4 = analymoms_type4(tilnu_21, tilnu_31, tilnu_41, I_clipped[type4])
        analymoms[type4,0] = m2; analymoms[type4,1] = m3; analymoms[type4,2] = m4
    if np.any(type5):
        m2, m3, m4 = analymoms_type5(tilnu_21, tilnu_31,           I_clipped[type5])
        analymoms[type5,0] = m2; analymoms[type5,1] = m3; analymoms[type5,2] = m4
    if np.any(type6):
        m2, m3, m4 = analymoms_type6(tilnu_21, tilnu_31, tilnu_41, I_clipped[type6])
        analymoms[type6,0] = m2; analymoms[type6,1] = m3; analymoms[type6,2] = m4
    
    return analymoms





def target_moms(tilnu21_arr: np.ndarray, tilnu31_arr: np.ndarray, tilnu41_arr: np.ndarray, I):
    """
    Produces matrix describing the desired central moments,
    used to compare with 

    t_moms.shape = (len(tilnu), I, 3)

    All the first moments are in t_moms[:,:,0]
    The moments with three given tilnus and 
    """
    tilnu21_arr = np.asarray(tilnu21_arr, float).ravel()
    tilnu31_arr = np.asarray(tilnu31_arr, float).ravel()
    tilnu41_arr = np.asarray(tilnu41_arr, float).ravel()

    tilnu42_arr = 3 * tilnu21_arr ** 2
    

    t_moms = np.empty((len(tilnu21_arr),len(I),3))

    t_moms[:,:,0] = np.outer(tilnu21_arr,I)
    t_moms[:,:,1] = np.outer(tilnu31_arr,I)
    t_moms[:,:,2] = np.outer(tilnu41_arr,I) + np.outer(tilnu42_arr,I**2)

    return t_moms

def make_analymoms_mat(tilnu21_arr: np.ndarray, tilnu31_arr: np.ndarray, tilnu41_arr: np.ndarray, I):
    
    
    tilnu21_arr = np.asarray(tilnu21_arr, float).ravel()
    tilnu31_arr = np.asarray(tilnu31_arr, float).ravel()
    tilnu41_arr = np.asarray(tilnu41_arr, float).ravel()
    I_arr       = np.asarray(I          , float).ravel()
    
    if not (len(tilnu21_arr) == len(tilnu31_arr) == len(tilnu41_arr)):
        raise RuntimeError("tilnu arrays need to be equal length")

    analymoms_mat = np.empty((tilnu21_arr.size,I_arr.size,3))

    for i in range(len(tilnu21_arr)):
        analymoms_mat[i,:,:]  = moms_from_Pearson(tilnu21_arr[i],tilnu31_arr[i],tilnu41_arr[i],I_arr)
    
    return analymoms_mat

def make_err_mat(tilnu21_arr: np.ndarray, tilnu31_arr: np.ndarray, tilnu41_arr: np.ndarray, I):

    t_moms = target_moms(tilnu21_arr, tilnu31_arr, tilnu41_arr, I)

    analy_moms = make_analymoms_mat(tilnu21_arr, tilnu31_arr, tilnu41_arr, I)

    err_moms = ( t_moms - analy_moms ) / np.maximum(t_moms,1e-12)

    return err_moms



# ============================================================
# ============================================================

# Example

# ============================================================
# ============================================================

#
# tilnuplx's and the row of intensities correspond to all the tilnus that make_err_mat will explore.
# In this case, make_err_mat will iterate over 11^3 x 5 combinations of tilnus and intensities.

tilnu21x = np.arange(1, 102, 10, dtype=int) # tilnu21x needs to be row matrix.
tilnu31x = np.arange(1, 102, 10, dtype=int) # The lengths of these matrices need to be equal.
tilnu41x = np.arange(1, 102, 10, dtype=int)

# Create meshgrid of tilnus to iterate over. Creates 11x11x11 matrix.

tilnu21mesh, tilnu31mesh, tilnu41mesh = np.meshgrid(tilnu21x,tilnu31x, tilnu41x, indexing="ij")

# The three 11x11x11 matrices are flattened into a 1d 1331 element array to be fed into make_err_mat.

tilnu21_arr = tilnu21mesh.ravel() 
tilnu31_arr = tilnu31mesh.ravel() 
tilnu41_arr = tilnu41mesh.ravel() 

I = np.array([5., 100., 1000., 10000., 100000.]) # Needs to be a row matrix.

err_mat = make_err_mat(tilnu21_arr, tilnu31_arr, tilnu41_arr, I)

np.save("analy_errormat.npy",err_mat) # complete error matrix saved as numpy array in case we need to access it again
np.savetxt("analy_mu2errs.txt",err_mat[:,:,0], fmt="%.2f")
np.savetxt("analy_mu3errs.txt",err_mat[:,:,1], fmt="%.2f")
np.savetxt("analy_mu4errs.txt",err_mat[:,:,2], fmt="%.2f")


'''
err_mat[:,:,0] contains the relative error of all the 2nd moments in our test. The columns
correspond to the intensities (five columns five intensities).

Just looking at the first column, going down, we increase tilnu21

so the first column first row corresponds to 
til21=1, til31=1, til41=1

second line: til21= 11, til31=1,til41=1
...
til21 = 1, til31=11, til41=1
til21 = 11, til31=11, til41=1
...
etc.

There is a "1.0" (100% relative error) at line 59. So if you want to know which
tilnu that corresponds to:

print(tilnu21_arr[59])
print(tilnu31_arr[59])
print(tilnu41_arr[59])

'''