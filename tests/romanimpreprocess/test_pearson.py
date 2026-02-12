"""

Tests for the individual Pearson distributions taken from draw_with_tilnus.
Takes in tilnus and a single intensity, draws from the calculated Pearson
distribution, and checks that it matches the target moments.

"""

import numpy as np  # noqa: F401
import matplotlib.pyplot as plt
from scipy.special import beta as EB

from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import random_from_type1
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import random_from_type3
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import random_from_type4
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import random_from_type5  # noqa: F401
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import random_from_type6


def test_pearson1(tilnu_21, tilnu_31, tilnu_41, I_single, Nsamp=100_000, rng=None):

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3*tilnu_21**2 * I_single**2

    p1_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 1 sampler...)")
    print(f"\n...)")


    for i in range(Nsamp):
        print(i)
        p1_draws[i] = random_from_type1(tilnu_21, tilnu_31, tilnu_41, I_single, rng)

    mu2_mc = p1_draws.var(ddof=0)
    mu3_mc = np.mean((p1_draws - p1_draws.mean())**3)
    mu4_mc = np.mean((p1_draws - p1_draws.mean())**4)

    print(f"\nMonte-Carlo test with ({Nsamp:,} samples for Pearson Type 1)")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")


    return


def test_pearson3(tilnu_21, tilnu_31, I_single, Nsamp=100_000, rng=None):

    tilnu_41 = 1.5 * tilnu_31**2 / tilnu_21

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3*tilnu_21**2 * I_single**2

    p3_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 3 sampler...)")
    print(f"\n...)")
    
    for i in range(Nsamp):
        p3_draws[i] = random_from_type3(tilnu_21, tilnu_31, I_single, rng=rng)

    mu2_mc = p3_draws.var(ddof=0)
    mu3_mc = np.mean((p3_draws - p3_draws.mean())**3)
    mu4_mc = np.mean((p3_draws - p3_draws.mean())**4)

    print(f"\nMonte-Carlo test with ({Nsamp:,} samples for Pearson Type 3)")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    return

def test_pearson4(tilnu_21, tilnu_31, tilnu_41, I_single, Nsamp=100_000, rng=None):

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3*tilnu_21**2 * I_single**2

    p4_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 4 sampler...)")
    print(f"\n...)")
    
    for i in range(Nsamp):
        print('Sample number ',i)
        p4_draws[i] = random_from_type4(tilnu_21, tilnu_31, tilnu_41, I_single, rng=rng)

    mu2_mc = p4_draws.var(ddof=0)
    mu3_mc = np.mean((p4_draws - p4_draws.mean())**3)
    mu4_mc = np.mean((p4_draws - p4_draws.mean())**4)

    print(f"\nMonte-Carlo test with ({Nsamp:,} samples for Pearson Type 4)")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    return


def test_pearson5(tilnu_21, tilnu_31, I_single, Nsamp=100_000, rng=None):

    def solve_tilnu41(tilnu21, tilnu31, I):
        """
        Closed-form solution:

            tilnu41 =
            [ 6 I ν21² ( −8 I ν21³ + 7 ν31² +
            I ν21³ ( 4 + ν31² / (I ν21³) )^(3/2) ) ]
            ------------------------------------------------
                    32 I ν21³ − ν31²
        """
        num  = -8 * I * tilnu21**3 + 7 * tilnu31**2
        root = (4 + tilnu31**2 / (I * tilnu21**3)) ** 1.5
        num += I * tilnu21**3 * root
        num *= 6 * I * tilnu21**2
        den  = 32 * I * tilnu21**3 - tilnu31**2
        return num / den

    tilnu_41 = solve_tilnu41(tilnu_21,tilnu_31,I_single)
    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3*tilnu_21**2 * I_single**2

    p5_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 5 sampler...)")
    print(f"\n...)")
    
    for i in range(Nsamp):
        p5_draws[i] = random_from_type5(tilnu_21, tilnu_31, I_single, rng)

    mu2_mc = p5_draws.var(ddof=0)
    mu3_mc = np.mean((p5_draws - p5_draws.mean())**3)
    mu4_mc = np.mean((p5_draws - p5_draws.mean())**4)

    print(f"\nMonte-Carlo test with ({Nsamp:,} samples for Pearson Type 5)")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    return

def test_pearson6(tilnu_21, tilnu_31, tilnu_41, I_single, Nsamp=100_000, rng=None):

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3*tilnu_21**2 * I_single**2

    p4_draws = np.empty(Nsamp)

    print(f"\nGenerating draws from Pearson 6 sampler...)")
    print(f"\n...)")
    
    for i in range(Nsamp):
        print(i)
        p4_draws[i] = random_from_type6(tilnu_21, tilnu_31, tilnu_41, I_single, rng)

    mu2_mc = p4_draws.var(ddof=0)
    mu3_mc = np.mean((p4_draws - p4_draws.mean())**3)
    mu4_mc = np.mean((p4_draws - p4_draws.mean())**4)

    print(f"\nMonte-Carlo test with ({Nsamp:,} samples for Pearson Type 4)")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    return





# EXAMPLE TESTS


#test_pearson1(tilnu_21 = 1.4375, tilnu_31 = -0.5, tilnu_41 = 0.15, I_single = 2.0, Nsamp=100_000)
#test_pearson3(tilnu_21 = 1.4375, tilnu_31 = -0.5, I_single = 2.0,                  Nsamp=100_000)
#test_pearson4(tilnu_21 = 5.0, tilnu_31 = -1.0, tilnu_41 = 5.0,   I_single = 50.0, Nsamp=100_000)
#test_pearson5(tilnu_21 = 7.0, tilnu_31 = -0.5, I_single = 120.0,                 Nsamp=100_000)
#test_pearson6(tilnu_21 = 6.0, tilnu_31 = -1.0, tilnu_41 = 0.3,   I_single = 3.5, Nsamp=100_000)