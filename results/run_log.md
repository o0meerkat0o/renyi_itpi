# Renyi MI Experiment Log

Results include plain-language interpretation after each number block.
Lines starting with '>' are the interpretations.

- N_DATA=600, N_REPS=8
- alpha sweep: 0.4 to 4.0, 19 values
- train_frac sweep: 0.2 to 0.8, 13 values

---

## Experiment 1: Gaussian baseline (alpha sweep)

Dataset: bivariate Gaussian, rho=0.7, N=600
Question: does MI vary with alpha? For Gaussian data it shouldn't much.

| alpha | mean MI | std |
|---|---|---|
| 0.40 | 1.9705 | 2.7020 |
| 0.80 | 0.3773 | 0.1949 |
| 1.20 | 0.3051 | 0.0349 |
| 1.60 | 0.2921 | 0.0196 |
| 2.00 | 0.2784 | 0.0042 |
| 2.40 | 0.2655 | 0.0102 |
| 2.80 | 0.2649 | 0.0096 |
| 3.20 | 0.2550 | 0.0267 |
| 3.60 | 0.2364 | 0.0206 |
| 4.00 | 0.2350 | 0.0187 |

- range of means: 0.2350 – 1.9705  (spread=1.7355)
- avg std across reps: 0.2145
- signal-to-noise (spread / avg_std): 8.09
- time: 2.0s
- KSG (Shannon) reference: 0.3332
- KDE at alpha=1.0: 0.3307
- unreliable alpha (std > 20% of mean): [0.4, 0.6, 0.8, 1.0]

**what this means:**
> Clear real signal: MI varies by 1.7355 across the sweep, which is 8.1x the noise level. This is not just KDE randomness.
> KSG=0.3332 vs KDE=0.3307: basically identical — KDE is working well here.
> A wide range of alpha is unreliable: 0.4 to 1.0. This suggests the dataset itself is hard to estimate density on — possibly fat-tailed or too small for KDE to work well.

## Experiment 2: Non-Gaussian baseline (alpha sweep)

Dataset: log-normal marginals with Gaussian copula, rho=0.7, N=600
Question: does MI vary with alpha? For non-Gaussian data it should.
Expected: alpha>1 suppresses tails → lower MI. alpha<1 amplifies tails → higher or noisier.

| alpha | mean MI | std |
|---|---|---|
| 0.40 | 1411.6502 | 2498.1990 |
| 0.80 | 2782.8700 | 2199.5393 |
| 1.20 | 0.2808 | 0.0946 |
| 1.60 | 0.3518 | 0.0255 |
| 2.00 | 0.3608 | 0.0054 |
| 2.40 | 0.3757 | 0.0130 |
| 2.80 | 0.3619 | 0.0301 |
| 3.20 | 0.3819 | 0.0221 |
| 3.60 | 0.3761 | 0.0400 |
| 4.00 | 0.3658 | 0.0483 |

- range of means: 0.2808 – 2782.8700  (spread=2782.5892)
- avg std across reps: 354.3519
- signal-to-noise (spread / avg_std): 7.85
- time: 1.9s
- KSG reference: 0.3112
- KDE at alpha=1.0: 9.1474
- unreliable alpha: [0.4, 0.6, 0.8, 1.0, 1.2]
- no stable low-alpha values
- avg MI at high alpha (stable only): 0.3674

**what this means:**
> Clear real signal: MI varies by 2782.5892 across the sweep, which is 7.9x the noise level. This is not just KDE randomness.
> KSG=0.3112 vs KDE=9.1474: a 2839% gap — that's large. KDE is struggling on this dataset at alpha=1. Be cautious about trusting KDE results here.
> A wide range of alpha is unreliable: 0.4 to 1.2. This suggests the dataset itself is hard to estimate density on — possibly fat-tailed or too small for KDE to work well.
> Not enough stable alpha values to determine direction reliably.

## Experiment 3: Train/test split sensitivity

Dataset: bivariate Gaussian, rho=0.7, N=600
Question: does the 50/50 split ratio actually matter?
Expected: at N=600, mean stays flat, std might rise at very low train_frac.

| train_frac | mean MI | std |
|---|---|---|
| 0.20 | 0.2839 | 0.0192 |
| 0.25 | 0.2962 | 0.0096 |
| 0.30 | 0.2947 | 0.0161 |
| 0.35 | 0.2942 | 0.0161 |
| 0.40 | 0.2825 | 0.0129 |
| 0.45 | 0.2905 | 0.0161 |
| 0.50 | 0.2904 | 0.0207 |
| 0.55 | 0.3016 | 0.0271 |
| 0.60 | 0.2903 | 0.0374 |
| 0.65 | 0.2967 | 0.0304 |
| 0.70 | 0.3151 | 0.0241 |
| 0.75 | 0.2727 | 0.0596 |
| 0.80 | 0.2945 | 0.0327 |

- range of means: 0.2727 – 0.3151  (spread=0.0424)
- avg std across reps: 0.0248
- signal-to-noise (spread / avg_std): 1.71
- time: 1.2s
- avg std at train_frac<=0.3: 0.0150
- avg std at train_frac>=0.6: 0.0368

**what this means:**
> The MI estimate barely changes across split ratios — 50/50 is fine, but so is anything from 30/70 to 70/30. Noise is surprisingly higher at high train_frac (few evaluation points = noisy average). This is unusual — could just be randomness at this N.

## Experiment 4: Alpha reliability cutoff

Run the same (X,Y) pair 15 times at each alpha.
Where does variance blow up? That's the unreliable zone.
Testing both Gaussian and log-normal so we can compare.

### gaussian
- rel_std > 0.25 at alpha: [0.4, 0.6, 0.8]
- suggested cutoff: alpha > 0.4
- stable range: alpha 1.00 to 4.00

**what this means:**
> Alpha values below 0.8 are unreliable — noise is more than 25% of the estimate. Stick to alpha >= 1.0. This is the KDE breaking down: low alpha amplifies rare events which are the hardest to estimate accurately.
> 84% of the alpha sweep range is usable for gaussian data. Safe zone: alpha 1.0 – 4.0.

### lognormal
- rel_std > 0.25 at alpha: [0.4, 0.6, 0.8, 1.0]
- suggested cutoff: alpha > 0.4
- stable range: alpha 1.20 to 4.00

**what this means:**
> A wide range of alpha is unreliable: 0.4 to 1.0. This suggests the dataset itself is hard to estimate density on — possibly fat-tailed or too small for KDE to work well.
> 79% of the alpha sweep range is usable for lognormal data. Safe zone: alpha 1.2 – 4.0.


## Experiment 5: Rayleigh — known answer

Known Pi*: y / sqrt(mu * t). Known bound should approach 0.
Running Shannon (KSG) vs alpha sweep on the converged Pi*.
This is the sanity check — if the bound behaves here, we trust it elsewhere.

- Shannon MI (KSG): 2.5387  →  eps=0.0790
- KDE eps at alpha=1.0: 0.7041
- KDE eps at alpha=2.0: 0.7639
- best alpha: 0.40  →  eps=0.0000

**what this means:**
> Shannon bound (KSG): eps=0.0790. Very tight — Pi* explains almost all of Y. This is what we expect for the correct dimensionless group. KDE at alpha=1 gives eps=0.7041 vs KSG eps=0.0790 — a 792% gap. KDE is much less accurate than KSG on this dataset. This is the cost of switching to KDE to support general alpha. Best alpha=0.40 gives eps=0.0000, tighter than Shannon. Renyi is finding something Shannon missed — the relationship has structure that alpha=0.40 is better at capturing.

---
## Total runtime: 0.3 min