"""
experiments/rayleigh.py

Test case: Rayleigh problem (velocity profile in a diffusing boundary layer).

Known answer: Pi* = y / sqrt(mu * t)
This is the standard test from the original IT-PI paper.

Running this at alpha=1 should reproduce the original IT-PI results.
Running at other alpha values lets you see how the bound changes —
though for Gaussian-ish data like this, the effect should be small.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

from buckingham_pi import calc_basis, create_labels
from itpi import run_itpi


def rayleigh_u(y, mu, U, t):
    """Analytical velocity profile: u = U * (1 - erf(y / (2*sqrt(mu*t))))"""
    return U * (1 - erf(y / (2 * np.sqrt(mu * t))))


def make_dataset(seed=42):
    np.random.seed(seed)
    rows = []
    for U in np.random.uniform(0.5, 1.0, 4):
        for mu in np.random.uniform(1e-3, 1e-2, 4):
            for y in np.linspace(0.02, 0.9, 8):
                for t in np.linspace(4, 10, 8):
                    rows.append([U, y, t, mu, rayleigh_u(y, mu, U, t)])
    data = np.array(rows)
    X = data[:, :4]
    Y = data[:, 4] / data[:, 0]  # normalize: Pi_o = u/U
    return X, Y


def main():
    X, Y = make_dataset()
    variables = ['U', 'y', 't', 'mu']

    # Dimensional matrix for [U, y, t, mu]
    # Rows = [length, time], columns = variables
    D = np.asarray([[1, 1, 0, 2],
                    [-1, 0, 1, -1]])
    nb = D.shape[1] - matrix_rank(D)
    basis = calc_basis(D, nb)

    print(f"dataset: N={len(X)}, num_basis={nb}")
    print(f"expected Pi*: y / sqrt(t * mu)\n")

    # Run at a few alpha values to compare
    alpha_values = [1.0, 0.5, 2.0, None]
    all_results = []

    for alpha in alpha_values:
        res = run_itpi(
            X=X, Y=Y,
            basis_matrices=basis,
            num_input=1,
            alpha=alpha,
            popsize=60,
            maxiter=800,
            seed=42,
        )
        coef = res['input_coef'][0]
        label = create_labels(np.array(coef), variables)[0]
        r0 = res['results'][0]
        a_str = f"{alpha:.1f}" if alpha is not None else "auto"
        all_results.append((a_str, label, r0))
        print(f"\nalpha={a_str}  Pi*: {label}")
        print(f"  eps_shannon={r0['eps_shannon']:.4f}  eps_alpha={r0['eps_alpha']:.4f}")

    # Plot: collapse for last run + eps vs alpha curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    pi_star = res['input_PI'][:, 0]
    pi_o = res['output_PI'].flatten()
    axes[0].scatter(pi_star, pi_o, alpha=0.4, s=8, c='steelblue')
    axes[0].set(xlabel=f'Pi*\n({label})', ylabel='u/U', title='Profile collapse (last run)')
    axes[0].grid(alpha=0.3)

    r0 = all_results[-1][2]
    if len(r0['a_curve']) > 1:
        axes[1].plot(r0['a_curve'], r0['e_curve'], 'steelblue', lw=2)
        axes[1].axvline(r0['alpha_used'], color='red', ls='--',
                        label=f"best alpha={r0['alpha_used']:.2f}")
        axes[1].axvline(1.0, color='gray', ls=':', label='alpha=1 (Shannon)')
        axes[1].legend(fontsize=9)
    axes[1].set(xlabel='alpha', ylabel='epsilon_lb', title='Bound vs alpha')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'rayleigh.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
