"""
experiments/sensitivity.py

Tests two open questions from Yi's feedback:

  1. Does the 50/50 train/test split ratio matter?
     Sweep train_frac from 0.3 to 0.8 and see if MI estimates change.
     Prediction: at N=1024 they probably won't, but variance might.

  2. Does using per-variable bandwidth (the fix we added) vs the old
     shared joint bandwidth actually change results?
     For 1D X and Y the effect should be small but measurable.

Run this before committing to any specific hyperparameter values.
Always write your predictions first, then compare.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from renyi_mi import renyi_mi


def make_test_data(N=500, seed=0):
    """Simple 1D Gaussian test case with known MI."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=N)
    Y = X + 0.5 * rng.normal(size=N)  # Y depends on X, some noise
    return X, Y


def sweep_train_frac(X, Y, alpha=1.5, fracs=None, n_reps=10):
    """
    For each train_frac, estimate renyi_mi n_reps times and record mean/std.
    High variance across reps = unstable. Big shift in mean = bias.
    """
    if fracs is None:
        fracs = np.linspace(0.2, 0.8, 13)

    means, stds = [], []
    for frac in fracs:
        vals = [renyi_mi(X, Y, alpha=alpha, train_frac=frac) for _ in range(n_reps)]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        print(f"  train_frac={frac:.2f}  mean={means[-1]:.4f}  std={stds[-1]:.4f}")

    return np.array(fracs), np.array(means), np.array(stds)


def sweep_alpha(X, Y, alphas=None, train_frac=0.5, n_reps=5):
    """
    See how MI estimate changes across alpha for this dataset.
    If X and Y are approximately Gaussian, it should be flat.
    If not, you'll see a trend — and the direction tells you something
    about what kind of dependence structure X and Y have.
    """
    if alphas is None:
        alphas = np.linspace(0.5, 4.0, 15)

    means, stds = [], []
    for a in alphas:
        vals = [renyi_mi(X, Y, alpha=a, train_frac=train_frac) for _ in range(n_reps)]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        print(f"  alpha={a:.2f}  mean={means[-1]:.4f}  std={stds[-1]:.4f}")

    return np.array(alphas), np.array(means), np.array(stds)


def main():
    X, Y = make_test_data(N=500)
    print(f"test data: N={len(X)}, 1D Gaussian with linear dependence\n")

    # --- Train/test split sensitivity ---
    print("=== train_frac sweep (alpha=1.5) ===")
    fracs, means_frac, stds_frac = sweep_train_frac(X, Y, alpha=1.5)

    # --- Alpha sensitivity ---
    print("\n=== alpha sweep (train_frac=0.5) ===")
    alphas, means_alpha, stds_alpha = sweep_alpha(X, Y, train_frac=0.5)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(fracs, means_frac, 'steelblue', lw=2, label='mean MI')
    axes[0].fill_between(fracs,
                         means_frac - stds_frac,
                         means_frac + stds_frac,
                         alpha=0.25, color='steelblue', label='±1 std')
    axes[0].axvline(0.5, color='gray', ls='--', label='default (0.5)')
    axes[0].set(xlabel='train_frac', ylabel='Renyi MI estimate',
                title='Sensitivity to train/test split\n(alpha=1.5, N=500)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(alphas, means_alpha, 'darkorange', lw=2, label='mean MI')
    axes[1].fill_between(alphas,
                         means_alpha - stds_alpha,
                         means_alpha + stds_alpha,
                         alpha=0.25, color='darkorange', label='±1 std')
    axes[1].axvline(1.0, color='gray', ls='--', label='alpha=1 (Shannon)')
    axes[1].set(xlabel='alpha', ylabel='Renyi MI estimate',
                title='MI vs alpha\n(Gaussian data: should be ~flat)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
