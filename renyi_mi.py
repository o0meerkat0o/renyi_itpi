"""
renyi_mi.py

Extends the original IT-PI mutual information estimator (KSG/KraskovMI1)
to support generalized Renyi MI of order alpha.

Why this exists:
  The original code is stuck at alpha=1 because KSG uses digamma functions
  that can't be generalized. To do arbitrary alpha you need actual density
  values (so you can raise p(x) to a power), which means switching to KDE.
  Every other difference from the original traces back to that one decision.

Approximation in use:
  True Renyi MI = H_alpha(Y) - H_alpha(Y|X)  [paper definition]
  We use:        H_alpha(X) + H_alpha(Y) - H_alpha(X,Y)  [additive form]
  These are equal at alpha=1 and for Gaussian data at any alpha.
  For non-Gaussian data and alpha != 1, this is an approximation.
  Most reliable in alpha ~ [0.5, 3]. Gets sketchier the further you go.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


def renyi_entropy(Z, alpha, bw=None, train_frac=0.5):
    """
    Renyi entropy of order alpha for a dataset Z, estimated via KDE.

    Z          : array of shape (N,) or (N, d)
    alpha      : order. alpha=1 gives Shannon entropy (via L'Hopital limit)
    bw         : bandwidth for KDE. if None, uses Scott's rule for Z's dimension
    train_frac : fraction of data used to FIT the KDE (rest used to EVALUATE it)

    Why train/test split at all?
      If you build a KDE on all the data and then check it at those same points,
      they'll always look high-density — you literally used them to define the
      distribution. At alpha=1 this bias roughly cancels in H(X)+H(Y)-H(XY).
      At alpha != 1 you're raising density to a power, so small overestimates
      get amplified and it doesn't cancel. Splitting fixes that.
    """
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = Z.shape[0]

    # Scott's rule: auto bandwidth based on Z's own dimension
    # This is the right call — use d+4 where d is THIS variable's dimension,
    # not the joint dimension. Marginals and joint each get their own bw.
    if bw is None:
        d = Z.shape[1]
        bw = n ** (-1.0 / (d + 4))

    idx = np.random.permutation(n)
    n_tr = max(int(n * train_frac), 10)

    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(Z[idx[:n_tr]])
    lp = kde.score_samples(Z[idx[n_tr:]])  # log p(z) for held-out points

    if abs(alpha - 1.0) < 1e-6:
        # Shannon entropy: -E[log p(z)]
        # L'Hopital gives this as the limit of Renyi as alpha -> 1
        return float(-np.mean(lp))

    # Renyi entropy of order alpha:
    #   H_alpha(Z) = (1/(1-alpha)) * log( E[p(z)^(alpha-1)] )
    #
    # In log space: log(p^(alpha-1)) = (alpha-1)*log(p) = (alpha-1)*lp
    # Then log(mean(exp(...))) via logsumexp to avoid overflow
    v = (alpha - 1.0) * lp
    vmax = v.max()
    log_mean = vmax + np.log(np.mean(np.exp(v - vmax)))
    return float(log_mean / (1.0 - alpha))


def renyi_mi(X, Y, alpha, train_frac=0.5):
    """
    Renyi MI of order alpha between X and Y.

    Uses the additive decomposition:
      I_alpha(X;Y) ≈ H_alpha(X) + H_alpha(Y) - H_alpha(X,Y)

    Each entropy term gets its own bandwidth (Scott's rule per dimension).
    This is better than using one shared bandwidth — X and Y live in smaller
    spaces than XY, so the joint bw would be too wide for the marginals.

    alpha > 1 : common/typical events dominate, rare events suppressed
    alpha = 1 : standard Shannon MI, matches KSG for large N
    alpha < 1 : rare/extreme events amplified (useful for fat-tailed data)

    For Gaussian data: result is the same for all alpha.
    For non-Gaussian data: result varies with alpha, and that variation
    is meaningful — it tells you which events are driving the relationship.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    XY = np.hstack([X, Y])

    # Each term gets its own bw based on its own dimension
    hx  = renyi_entropy(X,  alpha, train_frac=train_frac)
    hy  = renyi_entropy(Y,  alpha, train_frac=train_frac)
    hxy = renyi_entropy(XY, alpha, train_frac=train_frac)

    mi = hx + hy - hxy
    return float(max(mi, 0.0))  # clamp: MI can't be negative, but noise can push it there


def epsilon_lb(X, Y, alpha):
    """
    Irreducible error lower bound = exp(-I_alpha(X, Y)).

    0 = perfectly predictable (X tells you everything about Y)
    1 = completely uninformative

    Note: the full paper bound (Eq. 1) includes a c(alpha, p, h) factor
    that we don't compute here. So this is exp(-MI) only, not the complete bound.
    """
    return float(np.exp(-renyi_mi(X, Y, alpha)))
