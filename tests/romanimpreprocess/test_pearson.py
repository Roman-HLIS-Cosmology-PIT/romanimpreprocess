"""

Tests for the individual Pearson distributions taken from draw_with_tilnus.
Takes in tilnus and a single intensity, draws from the calculated Pearson
distribution. Mostly checks that the samplers produce draws.

Nsamp = 100_000 was chosen to keep the unit test brief. The variance of the third
and fourth moments require many more samples to converge towards the target.

"""

import time

import numpy as np  # noqa: F401
from romanimpreprocess.L1_to_L2.GalPoisson.draw_with_tilnus import (
    random_from_type1,
    random_from_type3,
    random_from_type4,
    random_from_type5,  # noqa: F401
    random_from_type6,
)


def _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.5):
    """Raises runtime error if MC moments and target moments have high relative error."""

    # relative errors (guard against divide-by-zero just in case)
    denom2 = max(abs(mu2_target), 1e-15)
    # denom3 = max(abs(mu3_target), 1e-15)

    rel2 = abs(mu2_mc - mu2_target) / denom2
    # rel3 = abs(mu3_mc - mu3_target) / denom3

    if rel2 > rtol:  # or (rel3 > rtol):
        raise RuntimeError(
            "Pearson MC 2nd moment check failed: "
            # f"relerr(mu2)={rel2:.3f}, relerr(mu3)={rel3:.3f} (rtol={rtol:.2f}). "
            f"relerr(mu2)={rel2:.3f}, (rtol={rtol:.2f}). "
            f"mu2_target={mu2_target:.6e}, mu2_mc={mu2_mc:.6e}; "
            f"mu3_target={mu3_target:.6e}, mu3_mc={mu3_mc:.6e}"
        )


def test_pearson1():
    """
    Executes an example test case of Pearson type 1 draws, verifying
    that the sampler produces draws, and verifying that the calculated
    2nd moment is close to the target 2nd moment.
    """

    tilnu_21 = 1.4375
    tilnu_31 = -0.5
    tilnu_41 = 0.15
    I_single = 2.0
    Nsamp = 100_000

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3 * tilnu_21**2 * I_single**2

    p1_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 1 sampler...")
    print("\n...")

    for i in range(Nsamp):
        p1_draws[i] = random_from_type1(tilnu_21, tilnu_31, tilnu_41, I_single, rng=None)

    print("\nDistribution sampled without issue.")

    mu2_mc = p1_draws.var(ddof=0)
    mu3_mc = np.mean((p1_draws - p1_draws.mean()) ** 3)
    mu4_mc = np.mean((p1_draws - p1_draws.mean()) ** 4)

    print(f"\nCalculated moments with {Nsamp:,} samples for Pearson Type 1:")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")
    print("Note: High samples needed to consistently get small relative tolerace for 3rd and 4th moments.")

    print("\nChecking relative error of 2nd moment...")

    _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.2)

    print("Check success.")

    return


def test_pearson3():
    """
    Executes an example test case of Pearson type 3 draws, verifying
    that the sampler produces draws, and verifying that the calculated
    2nd moment is close to the target 2nd moment.
    """

    tilnu_21 = 1.4375
    tilnu_31 = -0.5

    I_single = 2.0
    Nsamp = 100_000

    tilnu_41 = 1.5 * tilnu_31**2 / tilnu_21

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3 * tilnu_21**2 * I_single**2

    p3_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 3 sampler...")

    for i in range(Nsamp):
        p3_draws[i] = random_from_type3(tilnu_21, tilnu_31, I_single, rng=None)

    print("\nDistribution sampled without issue.")

    mu2_mc = p3_draws.var(ddof=0)
    mu3_mc = np.mean((p3_draws - p3_draws.mean()) ** 3)
    mu4_mc = np.mean((p3_draws - p3_draws.mean()) ** 4)

    print(f"\nMonte-Carlo test with {Nsamp:,} samples for Pearson Type 3")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    print("Note: High samples needed to consistently get small relative tolerace for 3rd and 4th moments.")

    print("\nChecking relative error of 2nd moment...")

    _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.2)

    print("Check success.")

    return


def test_pearson4_AR():
    """
    Executes an example test case of Pearson type 4 draws using the AR sampler, verifying
    that the sampler produces draws, and verifying that the calculated
    2nd moment is close to the target 2nd moment.
    """

    tilnu_21 = 5.0
    tilnu_31 = -1.0
    tilnu_41 = 5.0
    I_single = 50.0
    Nsamp = 100_000

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3 * tilnu_21**2 * I_single**2

    p4_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 4 AR sampler. This is the slowest sampler.")

    for i in range(Nsamp):
        p4_draws[i] = random_from_type4(tilnu_21, tilnu_31, tilnu_41, I_single, rng=None)

    print("\nDistribution sampled without issue.")

    mu2_mc = p4_draws.var(ddof=0)
    mu3_mc = np.mean((p4_draws - p4_draws.mean()) ** 3)
    mu4_mc = np.mean((p4_draws - p4_draws.mean()) ** 4)

    print(f"\nMonte-Carlo test with {Nsamp:,} samples for Pearson Type 4 using AR sampler")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    print("Note: High samples needed to consistently get small relative tolerace for 3rd and 4th moments.")

    print("\nChecking relative error of 2nd moment...")

    _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.2)

    print("Check success.")

    return


def test_pearson4_Devroye():
    """
    Executes an example test case of Pearson type 4 draws using the Devroye sampler, verifying
    that the sampler produces draws, and verifying that the calculated
    2nd moment is close to the target 2nd moment.
    """

    tilnu_21 = 1.0
    tilnu_31 = -1.0
    tilnu_41 = 10.0
    I_single = 3.0
    Nsamp = 100_000

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3 * tilnu_21**2 * I_single**2

    p4_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 4 Devroye sampler.")
    print("\nRuntimeError forced if samples don't complete in 100 seconds.")
    print("\n...")

    start_time = time.time()

    for i in range(Nsamp):
        if time.time() - start_time > 100:
            raise RuntimeError("Pearson 4 Devroye sampler exceeded 100 second limit")
        p4_draws[i] = random_from_type4(tilnu_21, tilnu_31, tilnu_41, I_single, rng=None)

    print("\nDistribution sampled without issue.")

    mu2_mc = p4_draws.var(ddof=0)
    mu3_mc = np.mean((p4_draws - p4_draws.mean()) ** 3)
    mu4_mc = np.mean((p4_draws - p4_draws.mean()) ** 4)

    print(f"\nMonte-Carlo test with {Nsamp:,} samples for Pearson Type 4 using Devroye sampler")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    print("Note: High samples needed to consistently get small relative tolerace for 3rd and 4th moments.")

    print("\nChecking relative error of 2nd moment...")

    _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.2)

    print("Check success.")

    return


def test_pearson5():
    """
    Executes an example test case of Pearson type 5 draws, verifying
    that the sampler produces draws, and verifying that the calculated
    2nd moment is close to the target 2nd moment.
    """

    tilnu_21 = 7.0
    tilnu_31 = -0.5

    I_single = 120.0
    Nsamp = 100_000

    def solve_tilnu41(tilnu21, tilnu31, I_single):
        """
        Closed-form solution:

            tilnu41 =
            [ 6 I ν21² ( −8 I ν21³ + 7 ν31² +
            I ν21³ ( 4 + ν31² / (I ν21³) )^(3/2) ) ]
            ------------------------------------------------
                    32 I ν21³ − ν31²
        """
        num = -8 * I_single * tilnu21**3 + 7 * tilnu31**2
        root = (4 + tilnu31**2 / (I_single * tilnu21**3)) ** 1.5
        num += I_single * tilnu21**3 * root
        num *= 6 * I_single * tilnu21**2
        den = 32 * I_single * tilnu21**3 - tilnu31**2
        return num / den

    tilnu_41 = solve_tilnu41(tilnu_21, tilnu_31, I_single)
    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3 * tilnu_21**2 * I_single**2

    p5_draws = np.empty(Nsamp)

    print(f"\nGenerating {Nsamp:,} draws from Pearson 5 sampler...")
    print("\n...")

    for i in range(Nsamp):
        p5_draws[i] = random_from_type5(tilnu_21, tilnu_31, I_single, rng=None)

    print("\nDistribution sampled without issue.")

    mu2_mc = p5_draws.var(ddof=0)
    mu3_mc = np.mean((p5_draws - p5_draws.mean()) ** 3)
    mu4_mc = np.mean((p5_draws - p5_draws.mean()) ** 4)

    print(f"\nMonte-Carlo test with {Nsamp:,} samples for Pearson Type 5")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    print("Note: High samples needed to consistently get small relative tolerace for 3rd and 4th moments.")

    print("\nChecking relative error of 2nd moment...")

    _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.2)

    print("Check success.")

    return


# test_pearson1(tilnu_21 = 1.4375, tilnu_31 = -0.5, tilnu_41 = 0.15, I_single = 2.0, Nsamp=100_000)
# test_pearson3(tilnu_21 = 1.4375, tilnu_31 = -0.5, I_single = 2.0,                  Nsamp=100_000)
# test_pearson4(tilnu_21 = 5.0, tilnu_31 = -1.0, tilnu_41 = 5.0,   I_single = 50.0, Nsamp=100_000)
# test_pearson5(tilnu_21 = 7.0, tilnu_31 = -0.5, I_single = 120.0,                 Nsamp=100_000)
# test_pearson6(tilnu_21 = 6.0, tilnu_31 = -1.0, tilnu_41 = 0.3,   I_single = 3.5, Nsamp=100_000)


def test_pearson6():
    """
    Executes an example test case of Pearson type 6 draws, verifying
    that the sampler produces draws, and verifying that the calculated
    2nd moment is close to the target 2nd moment.
    """

    tilnu_21 = 6.0
    tilnu_31 = -1.0
    tilnu_41 = 0.3
    I_single = 3.5
    Nsamp = 100_000

    # Define target moments
    mu2_target = tilnu_21 * I_single
    mu3_target = tilnu_31 * I_single
    mu4_target = tilnu_41 * I_single + 3 * tilnu_21**2 * I_single**2

    p4_draws = np.empty(Nsamp)

    print("\nGenerating draws from Pearson 6 sampler...")
    print("\n...")

    for i in range(Nsamp):
        p4_draws[i] = random_from_type6(tilnu_21, tilnu_31, tilnu_41, I_single, rng=None)

    print("\nDistribution sampled without issue.")

    mu2_mc = p4_draws.var(ddof=0)
    mu3_mc = np.mean((p4_draws - p4_draws.mean()) ** 3)
    mu4_mc = np.mean((p4_draws - p4_draws.mean()) ** 4)

    print(f"\nMonte-Carlo test with {Nsamp:,} samples for Pearson Type 6")
    print(f"  μ₂_target={mu2_target:.6e}  |  μ₂_calc={mu2_mc:.6e}  Δ={mu2_mc - mu2_target:+.3e}")
    print(f"  μ₃_target={mu3_target:.6e}  |  μ₃_calc={mu3_mc:.6e}  Δ={mu3_mc - mu3_target:+.3e}")
    print(f"  μ₄_target={mu4_target:.6e}  |  μ₄_calc={mu4_mc:.6e}  Δ={mu4_mc - mu4_target:+.3e}")

    print("Note: High samples needed to consistently get small relative tolerace for 3rd and 4th moments.")

    print("\nChecking relative error of 2nd moment...")

    _raise_if_moments_bad(mu2_mc, mu2_target, mu3_mc, mu3_target, rtol=0.2)

    print("Check success.")

    return


"""test_pearson1()
test_pearson3()
test_pearson4_AR()
test_pearson4_Devroye()
test_pearson5()
test_pearson6()"""
