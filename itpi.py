"""
itpi.py

IT-PI runner with generalized alpha support.

The original code (ALD-Lab/IT_PI) only works at alpha=1 because it uses
KSG (KraskovMI1), which is built around digamma functions that can't be
generalized to arbitrary alpha. This version:

  - Runs CMA-ES at alpha=1 internally (still using KSG, which is more
    stable and accurate for optimization than KDE)
  - After CMA-ES converges and Pi* is found, THEN evaluates the bound
    at whatever alpha you asked for (using KDE-based renyi_mi)

This is the key design decision: Pi* found at alpha=1 is empirically
close to the optimal Pi* at other alpha values, so we don't lose much
by decoupling the search from the bound computation.

alpha choices:
  None  -> auto-searches for the alpha that gives the tightest bound
  1.0   -> Shannon MI, identical to original IT-PI
  < 1   -> rare/extreme events matter more
  > 1   -> common events dominate, outliers suppressed
"""

import warnings
import random
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import psi
import scipy.spatial as scispa
from cma import CMAEvolutionStrategy

from buckingham_pi import calc_pi, calc_pi_omega, create_labels
from renyi_mi import renyi_mi, epsilon_lb


# ---- Shannon MI via KSG (original estimator, alpha=1 only) ----------------

def kraskov_mi(x, y, k=5):
    """
    Shannon MI via KSG (Kraskov estimator, k-nearest-neighbors).
    This is the original IT-PI estimator. Kept here because:
      1. It's more accurate than KDE at alpha=1 for finite N
      2. CMA-ES runs more stably with it during optimization
    Not used for the final Renyi bound — that's handled by renyi_mi.py.
    """
    V = np.hstack([x, y])
    kdtree = scispa.KDTree(V)
    ei, _ = kdtree.query(V, k + 1, p=np.inf)
    dM = ei[:, -1]
    nx = scispa.KDTree(x).query_ball_point(x, dM, p=np.inf, return_length=True)
    ny = scispa.KDTree(y).query_ball_point(y, dM, p=np.inf, return_length=True)
    N = x.shape[0]
    return float(psi(k) - (psi(nx) + psi(ny)).mean() + psi(N))


# ---- Main runner -----------------------------------------------------------

def run_itpi(
    X,
    Y,
    basis_matrices,
    num_input=1,
    alpha=None,
    train_frac=0.5,
    popsize=80,
    maxiter=1000,
    num_trials=5,
    p_norm=2,
    seed=None,
):
    """
    Find the best Pi group(s) for predicting Y from X, then compute
    the Renyi MI lower bound on irreducible prediction error.

    Parameters
    ----------
    X             : (N, num_vars) array of physical variables
    Y             : (N,) output variable (already nondimensionalized if needed)
    basis_matrices: output of calc_basis() — the Pi group null space
    num_input     : how many Pi groups to find (usually 1)
    alpha         : Renyi order. None = auto-search, 1.0 = Shannon (original)
    train_frac    : KDE train/test split fraction (0.5 = 50/50)
    popsize       : CMA-ES population size (use 300+ for production)
    maxiter       : CMA-ES max generations (use 50000+ for production)
    num_trials    : half-data resampling trials for uncertainty estimate
    p_norm        : Lp norm for prediction (2=MSE, 1=MAE)
    seed          : random seed for reproducibility

    Returns
    -------
    dict with keys: input_coef, input_PI, output_PI, results, alpha_mode
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    N = X.shape[0]
    num_basis = basis_matrices.shape[0]
    num_params = num_basis * num_input
    Y_col = Y.reshape(-1, 1)

    alpha_label = f"alpha={alpha}" if alpha is not None else "auto-search"
    print("=" * 55)
    print(f"IT-PI  |  {alpha_label}")
    print(f"N={N}, popsize={popsize}, maxiter={maxiter}")
    print("=" * 55)

    # CMA-ES objective: always uses Shannon MI (KSG) for stability
    # The user's alpha only matters when we compute the final bound below
    def safe_obj(params):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                pi_list = [
                    calc_pi(tuple(params[i * num_basis:(i + 1) * num_basis]),
                            basis_matrices, X)
                    for i in range(num_input)
                ]
                Pi = np.column_stack(pi_list)
            except Exception:
                return random.uniform(1e4, 1e6)
        if np.any(~np.isfinite(Pi)):
            return random.uniform(1e4, 1e6)
        return -kraskov_mi(Pi, Y_col, k=5)

    # Run CMA-ES
    options = {
        'bounds': [[-2] * num_params, [2] * num_params],
        'maxiter': maxiter,
        'tolx': 1e-6,
        'tolfun': 1e-6,
        'popsize': popsize,
        'verbose': -9,
        'seed': seed if seed else random.randint(0, 9999),
    }
    es = CMAEvolutionStrategy([0.1] * num_params, 0.5, options)
    gen = 0
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [safe_obj(p) for p in sols])
        gen += 1
        if gen % 200 == 0:
            print(f"  gen {gen:4d}  best_MI={-es.result.fbest:.4f}")

    print(f"done. {gen} generations.")

    # Extract Pi* from best CMA-ES solution
    best_params = es.result.xbest
    a_list = [tuple(best_params[i * num_basis:(i + 1) * num_basis])
              for i in range(num_input)]
    coef_list = [np.sum(np.array(a).reshape(-1, 1, 1) * basis_matrices, axis=0)
                 for a in a_list]
    norm_coefs = [np.round(c / np.max(np.abs(c)), 2) for c in coef_list]
    input_PI = np.column_stack([
        calc_pi_omega(np.array(c), X) for c in norm_coefs
    ])

    # Compute bounds
    print("\ncomputing bounds...")
    results = []
    cols = list(range(input_PI.shape[1]))
    if input_PI.shape[1] > 1:
        cols.append('all')

    alpha_min = 1.0 / (1.0 + p_norm) + 1e-4  # lower bound from paper

    for j in cols:
        xi = input_PI if j == 'all' else input_PI[:, j:j + 1]
        lbl = 'Pi* joint' if j == 'all' else f'Pi_{j + 1}*'

        # Shannon bound (alpha=1) — using KSG for a fair comparison to original
        mi_s = kraskov_mi(xi, Y_col, k=5)
        eps_s = float(np.exp(-max(mi_s, 0.0)))

        # Uncertainty: half-data resampling
        eps_halves = []
        for _ in range(num_trials):
            idx = np.random.choice(N, N // 2, replace=False)
            mi_h = kraskov_mi(xi[idx], Y_col[idx], k=5)
            eps_halves.append(np.exp(-max(mi_h, 0.0)))
        uq = abs(eps_s - np.mean(eps_halves))

        # Renyi bound at user's alpha
        if alpha is not None:
            eps_user = epsilon_lb(xi, Y_col, alpha)
            alpha_used = alpha
            a_curve = np.array([alpha])
            e_curve = np.array([eps_user])
        else:
            # Sweep alpha then refine with bounded minimization
            a_sweep = np.linspace(alpha_min, 5.0, 20)
            e_sweep = np.array([epsilon_lb(xi, Y_col, a) for a in a_sweep])
            result = minimize_scalar(
                lambda a: epsilon_lb(xi, Y_col, a),
                bounds=(alpha_min, 10.0),
                method='bounded',
                options={'xatol': 0.05},
            )
            alpha_used = float(result.x)
            eps_user = float(result.fun)
            a_curve = a_sweep
            e_curve = e_sweep

        print(f"  {lbl}:  Shannon={eps_s:.4f}±{uq:.4f}  "
              f"alpha={alpha_used:.2f} -> eps={eps_user:.4f}  "
              f"delta={eps_s - eps_user:+.4f}")

        results.append({
            'label': lbl,
            'eps_shannon': eps_s,
            'eps_alpha': eps_user,
            'alpha_used': alpha_used,
            'uq': uq,
            'a_curve': a_curve,
            'e_curve': e_curve,
        })

    return {
        'input_coef': norm_coefs,
        'input_PI': input_PI,
        'output_PI': Y_col,
        'results': results,
        'alpha_mode': alpha_label,
    }
