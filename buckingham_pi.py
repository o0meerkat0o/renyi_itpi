"""
buckingham_pi.py

Buckingham Pi / dimensional analysis utilities.
Unchanged from the original IT-PI repo (ALD-Lab/IT_PI).
Kept separate so it's easy to swap in updates from upstream.
"""

import numpy as np
from numpy.linalg import inv, matrix_rank


def calc_basis(D_in, col_range):
    """Compute the null-space basis vectors of the dimensional matrix D."""
    D_in = np.matrix(D_in)
    num_rows = np.shape(D_in)[0]
    Din1 = D_in[:, :num_rows]
    Din2 = D_in[:, num_rows:]

    basis_matrices_list = []
    for i in range(col_range):
        x2 = np.zeros((col_range, 1))
        x2[i, 0] = -1
        x1_matrix = -inv(Din1) * Din2 * np.matrix(x2)
        basis_matrices_list.append(np.vstack((np.asarray(x1_matrix), np.asarray(x2))))

    return np.asarray(basis_matrices_list)


def calc_pi(c, basis_matrices, X):
    """Compute Pi group from coefficient vector c, basis matrices, and data X."""
    coef_pi = np.sum(np.array(c).reshape(-1, 1, 1) * basis_matrices, axis=0)
    pi_mat = np.ones((X.shape[0], 1))
    for i in range(X.shape[1]):
        pi_mat = np.multiply(pi_mat, (X[:, i] ** coef_pi[i, 0]).reshape(-1, 1))
    return pi_mat


def calc_pi_omega(coef_pi, X):
    """Same as calc_pi but takes a precomputed coefficient array directly."""
    pi_mat = np.ones((X.shape[0], 1))
    for i in range(X.shape[1]):
        pi_mat = np.multiply(pi_mat, (X[:, i] ** coef_pi[i, 0]).reshape(-1, 1))
    return pi_mat


def create_labels(omega, variables):
    """
    Turn exponent arrays into readable Pi group strings.
    e.g. [1, -0.5, -0.5] with ['y','t','mu'] -> 'y^1.0 / (t^0.5 * mu^0.5)'
    """
    labels = []
    for row in omega:
        pos, neg = '', ''
        for i, val in enumerate(row):
            val = float(np.round(val, 2))
            if val > 0:
                term = f"{variables[i]}^{{{val}}}"
                pos = term if not pos else pos + f" * {term}"
            elif val < 0:
                term = f"{variables[i]}^{{{-val}}}"
                neg = term if not neg else neg + f" * {term}"
        if not neg:
            labels.append(pos)
        elif not pos:
            labels.append(f"1 / ({neg})")
        else:
            labels.append(f"({pos}) / ({neg})")
    return labels
