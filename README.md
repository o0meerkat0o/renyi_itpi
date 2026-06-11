# renyi-itpi

Extension of [IT-PI](https://github.com/ALD-Lab/IT_PI) (Yuan & Lozano-Durán 2025) to support generalized Rényi mutual information of order α.

The original code only works at α=1 (Shannon MI) because it uses KSG, which relies on digamma functions that can't be generalized. This repo adds support for arbitrary α by switching to KDE-based density estimation for the final bound computation.

---

## What alpha does

α controls which events drive the MI estimate when measuring how informative a Pi group is about the output:

| α | Effect |
|---|--------|
| < 1 | rare/extreme events amplified — use for fat-tailed data or worst-case prediction |
| = 1 | standard Shannon MI, identical to original IT-PI |
| > 1 | common events dominate, outliers suppressed — use for average-case prediction |
| `None` | auto-searches for the α that gives the tightest bound |

For Gaussian data, the result is the same for all α. The variation only shows up in non-Gaussian distributions, and understanding *why* it changes is part of the point.

---

## Approximations and known limitations

**Additive decomposition:**
We use `I_α(X;Y) ≈ H_α(X) + H_α(Y) - H_α(X,Y)` instead of the paper's exact definition `H_α(Y) - H_α(Y|X)`. These are equal at α=1 and for Gaussian data. For non-Gaussian data with α ≠ 1, it's an approximation. The exact version would require density estimation inside another density estimation, which is impractical at our data sizes.

**KDE vs KSG:**
KSG (used in original) is more accurate than KDE for Shannon MI at finite N. We still use KSG *inside* CMA-ES for optimization stability. KDE is only used for the final Rényi bound.

**Train/test split:**
KDE is fit on 50% of data and evaluated on the other 50% to avoid a bias that gets amplified when you raise density to a power (α ≠ 1). This makes estimates noisier. The split ratio is configurable.

**Missing c(α, p, h) factor:**
Equation 1 of the paper includes a prefactor `c(α, p, h_α_o)` that we don't compute. So `epsilon_lb` gives `exp(-I_α)`, not the complete bound.

**Reliable range:** α ∈ [0.5, 3]. Results get sketchier further out because you're raising density estimates to higher powers.

---

## Structure

```
renyi_itpi/
├── renyi_mi.py          # renyi_entropy + renyi_mi + epsilon_lb
├── itpi.py              # CMA-ES runner with alpha support
├── buckingham_pi.py     # dimensional analysis utils (unchanged from original)
├── experiments/
│   ├── rayleigh.py      # standard test case, known answer: y/sqrt(mu*t)
│   └── sensitivity.py   # sweep train_frac and alpha, check stability
├── tests/
│   └── test_renyi_mi.py # sanity checks
├── results/             # output plots go here
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# run the Rayleigh test case
python experiments/rayleigh.py

# check sensitivity to hyperparameters
python experiments/sensitivity.py

# run tests
python tests/test_renyi_mi.py
```
---

## Relation to original IT-PI

Setting `alpha=1.0` in `run_itpi()` reproduces the original behavior (CMA-ES + KSG, same Pi* discovery). The only difference at α=1 is that the final bound uses KDE instead of KSG, which is slightly less accurate — this is the price of generality.

The original repo is at [ALD-Lab/IT_PI](https://github.com/ALD-Lab/IT_PI).
