"""
tests/test_renyi_mi.py

Sanity checks for renyi_mi and renyi_entropy.

Things we want to verify:
  1. At alpha=1, result is close to KSG (Shannon MI) — not identical bc
     KDE is less accurate than KSG, but should be in the right ballpark
  2. For Gaussian data, MI doesn't change much across alpha (Gaussians are
     "alpha-invariant" — the Renyi MI equals Shannon MI for all alpha)
  3. MI is non-negative
  4. Independent variables give MI near zero
  5. More dependent variables give higher MI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from renyi_mi import renyi_mi, renyi_entropy
from itpi import kraskov_mi


N = 800
rng = np.random.default_rng(0)


def make_dependent(rho=0.9, n=N):
    """Bivariate Gaussian with correlation rho."""
    cov = [[1, rho], [rho, 1]]
    data = rng.multivariate_normal([0, 0], cov, size=n)
    return data[:, 0], data[:, 1]


def make_independent(n=N):
    return rng.normal(size=n), rng.normal(size=n)


def test_non_negative():
    X, Y = make_dependent()
    for alpha in [0.5, 1.0, 1.5, 2.5]:
        mi = renyi_mi(X, Y, alpha)
        assert mi >= 0, f"MI negative at alpha={alpha}: {mi}"


def test_independent_near_zero():
    X, Y = make_independent()
    for alpha in [0.5, 1.0, 2.0]:
        mi = renyi_mi(X, Y, alpha)
        assert mi < 0.15, f"MI for independent vars too high at alpha={alpha}: {mi}"


def test_dependent_higher_than_independent():
    Xd, Yd = make_dependent(rho=0.8)
    Xi, Yi = make_independent()
    mi_dep = renyi_mi(Xd, Yd, alpha=1.0)
    mi_ind = renyi_mi(Xi, Yi, alpha=1.0)
    assert mi_dep > mi_ind, f"dependent MI ({mi_dep:.4f}) not > independent MI ({mi_ind:.4f})"


def test_alpha1_close_to_ksg():
    """
    KDE at alpha=1 should be in the same ballpark as KSG.
    KDE is noisier, so we allow a reasonable tolerance.
    This tolerance might need to be loosened for smaller N.
    """
    X, Y = make_dependent(rho=0.7)
    X2d = X.reshape(-1, 1)
    Y2d = Y.reshape(-1, 1)

    mi_kde = renyi_mi(X, Y, alpha=1.0)
    mi_ksg = kraskov_mi(X2d, Y2d, k=5)

    print(f"\nalpha=1: KDE={mi_kde:.4f}, KSG={mi_ksg:.4f}, diff={abs(mi_kde - mi_ksg):.4f}")
    assert abs(mi_kde - mi_ksg) < 0.4, (
        f"KDE and KSG too far apart: KDE={mi_kde:.4f}, KSG={mi_ksg:.4f}"
    )


def test_gaussian_alpha_invariant():
    """
    For Gaussian data, renyi_mi should be approximately the same across alpha.
    Checks that the range of MI values across alpha is small.
    """
    X, Y = make_dependent(rho=0.6)
    alphas = [0.6, 1.0, 1.5, 2.0, 3.0]
    mis = [renyi_mi(X, Y, alpha=a) for a in alphas]

    print(f"\nGaussian MI across alpha: {[f'{m:.4f}' for m in mis]}")
    spread = max(mis) - min(mis)
    # KDE noise means some spread is expected, but it shouldn't be huge
    assert spread < 0.5, f"MI varies too much across alpha for Gaussian data: spread={spread:.4f}"


def test_entropy_non_negative_shannon():
    Z = rng.normal(size=500)
    h = renyi_entropy(Z, alpha=1.0)
    # Differential entropy CAN be negative (for narrow distributions), just check it runs
    assert np.isfinite(h)


if __name__ == '__main__':
    # Run without pytest
    tests = [
        test_non_negative,
        test_independent_near_zero,
        test_dependent_higher_than_independent,
        test_alpha1_close_to_ksg,
        test_gaussian_alpha_invariant,
        test_entropy_non_negative_shannon,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
