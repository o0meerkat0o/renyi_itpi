# code notes — renyi_itpi

same style as the alpha notes. line by line, plain language, questions for Yi marked with ask:

---

## renyi_mi.py

this is the core new file. everything else either existed before or exists to support this.

---

### `renyi_entropy(Z, alpha, bw=None, train_frac=0.5)`

this is the building block. takes a dataset Z and returns its Renyi entropy of order alpha.
called three times per MI estimate: once for X, once for Y, once for XY together.

---

**`if Z.ndim == 1: Z = Z.reshape(-1, 1)`**
* [1,2,3] → [[1],[2],[3]], sklearn needs 2D input
* same as in the original genericAlpha notebook
* nothing to do with alpha, just shape fixing

---

**`d = Z.shape[1]` and `bw = n ** (-1.0 / (d + 4))`**
* Scott's rule — auto picks how wide to make the gaussian bumps in the KDE
* THIS is the fix from Yi's feedback. the original notebook used the joint dimension (dx+dy) for all three KDE calls
* now each call uses its own d — X uses its dimension, Y uses its dimension, XY uses dx+dy
* why this matters: if X is 1D and XY is 2D, the old code used bw = N^(-1/6) for X when it should use N^(-1/5). wider bandwidth = over-smoothed = density estimates are too flat
* over-smoothing on marginals inflates H(X) and H(Y) slightly, which messes with H(X)+H(Y)-H(XY)
* the further alpha is from 1 the worse this gets because you're raising density to a power
* ask: how much does this actually change the results on real data? run sensitivity.py and compare

---

**`idx = np.random.permutation(n)` and `n_tr = max(int(n * train_frac), 10)`**
* shuffle the data, then take the first train_frac fraction to fit the KDE
* rest is held out for evaluation
* why: if you fit and evaluate on the same points, those points always look high-density (you built the distribution around them). that's a bias.
* at alpha=1 this bias roughly cancels in H(X)+H(Y)-H(XY)
* at alpha≠1 you raise density to a power, so small overestimates get amplified, it doesn't cancel
* `max(..., 10)` is a safety floor — if you somehow have 15 data points and train_frac=0.5, don't try to fit a KDE on 7 points
* NOT in original IT-PI at all. KSG never evaluates density at training points so it doesn't have this problem
* ask: is 50/50 the right ratio? run the split sweep in sensitivity.py and see if results shift

---

**`kde = KernelDensity(kernel='gaussian', bandwidth=bw)`**
**`kde.fit(Z[idx[:n_tr]])`**
**`lp = kde.score_samples(Z[idx[n_tr:]])`**
* fit a gaussian KDE on the training half
* evaluate log p(z) at every held-out point
* `score_samples` returns log probabilities, so lp is an array of log p(z) values
* KDE puts a little gaussian bell curve at each training point, adds them all up, that's your density estimate
* switching to KDE is literally the only reason this whole file exists — KSG can't give you density values, only MI directly. KDE gives you density values so you can raise them to a power for Renyi

---

**`if abs(alpha - 1.0) < 1e-6: return float(-np.mean(lp))`**
* Shannon entropy = -E[log p(z)] = negative average of log probabilities
* use `abs(alpha - 1.0) < 1e-6` not `alpha == 1.0` because floating point. 1.0000000001 != 1.0 to a computer
* even at alpha=1 this is technically a different estimator than the original — KSG is designed to be unbiased for Shannon MI, KDE is not. so this version is slightly worse at alpha=1
* ask: how much worse? the test showed KDE=0.2954, KSG=0.3523 on the same data. that's a 16% gap. does that matter for our use case?

---

**`v = (alpha - 1.0) * lp`**
**`vmax = v.max()`**
**`log_mean = vmax + np.log(np.mean(np.exp(v - vmax)))`**
**`return float(log_mean / (1.0 - alpha))`**
* this is the actual new Renyi stuff. 100% not in original code.
* Renyi entropy formula: H_alpha(Z) = (1/(1-alpha)) * log(E[p(z)^(alpha-1)])
* in log space: log(p^(alpha-1)) = (alpha-1) * log(p) = (alpha-1) * lp → that's `v`
* want log(mean(exp(v))). but exp() on large numbers overflows
* logsumexp trick: pull out the max, subtract before exp-ing, add it back after. mathematically identical, doesn't blow up
* divide by (1-alpha) to finish the Renyi entropy formula
* the further alpha is from 1, the more this amplifies any errors in the density estimate
* ask: what's the actual blowup look like? run exp 4 (reliability) to find the cutoff

---

### `renyi_mi(X, Y, alpha, train_frac=0.5)`

wraps renyi_entropy. calls it three times and does H(X) + H(Y) - H(XY).

**`XY = np.hstack([X, Y])`**
* stick X and Y side by side to make the joint variable
* same thing KraskovMI1 does in the original code

**`hx = renyi_entropy(X, ...) hy = renyi_entropy(Y, ...) hxy = renyi_entropy(XY, ...)`**
* each one gets its own bw now (the per-variable fix)
* this is the approximation: we're computing I_alpha ≈ H_alpha(X) + H_alpha(Y) - H_alpha(XY)
* the paper's real definition is H_alpha(Y) - H_alpha(Y|X) which doesn't split this way
* we can't do the real version without estimating density inside density, which is basically impossible
* the approximation is exact at alpha=1 and exact for gaussian data at any alpha
* ask: how bad is the error for our actual physics data? probably fine if it's near-Gaussian, unknown otherwise

**`return float(max(mi, 0.0))`**
* MI can't be negative in theory. noise can push it slightly negative.
* clamp it. original code doesn't do this.

---

### `epsilon_lb(X, Y, alpha)`

* just exp(-MI). converts MI into the error bound.
* 0 = X perfectly predicts Y, 1 = X tells you nothing
* missing the c(alpha, p, h) factor from paper eq 1. so this is a partial bound.
* ask: how much does c() matter? is exp(-MI) alone in the right ballpark?

---

## itpi.py

wrapper that runs CMA-ES to find Pi*, then calls renyi_mi to compute the bound.
most of this is unchanged from original IT-PI.

---

### `kraskov_mi(x, y, k=5)`

* the original KSG estimator, copied from IT-PI
* builds a KD-tree over the joint space, finds k nearest neighbors for each point
* counts how many neighbors fall within that radius in X-space and Y-space separately
* digamma function correction gives an unbiased MI estimate
* this is more accurate than KDE at alpha=1, which is why we still use it inside CMA-ES
* NOT used for the final Renyi bound — just for optimization

---

### `run_itpi(...)` — the CMA-ES loop

**`safe_obj(params)`**
* the function CMA-ES is trying to minimize
* takes a set of exponent coefficients, computes Pi from them, measures MI between Pi and Y
* returns negative MI (CMA-ES minimizes, we want to maximize MI)
* if Pi has infinite or NaN values (e.g. negative base raised to fractional exponent), returns a huge penalty number instead of crashing
* always uses KSG at alpha=1 here, regardless of what alpha the user asked for
* why: KDE is noisier than KSG. during optimization you're evaluating hundreds of candidate Pi groups. noisy MI estimates make the optimization landscape rough and CMA-ES struggles. KSG is smooth.
* the user's alpha only affects the final bound, after Pi* is already found

**`es = CMAEvolutionStrategy([0.1] * num_params, 0.5, options)`**
* CMA-ES: evolutionary optimizer, kind of like gradient descent but doesn't need gradients
* starts with a population of candidate exponent vectors, each one defines a Pi group
* measures MI for all of them, keeps the good ones, throws out the bad ones, mutates
* `[0.1] * num_params` = starting point (small exponents)
* `0.5` = initial step size (how much to vary the candidates)
* runs until it stops finding improvements or hits maxiter

**extracting Pi* after CMA-ES:**
* `es.result.xbest` = the exponent vector that gave the highest MI
* convert to actual coefficient array, normalize so the largest exponent is 1
* plug back into the data to get the actual Pi* values

**computing the bounds:**
* Shannon bound: just runs KSG on Pi* vs Y, takes exp(-MI). this is the original IT-PI result.
* uncertainty: re-runs KSG on random half-subsets of data. the gap between full-data and half-data estimates tells you how stable the result is
* Renyi bound: if alpha is given, just calls epsilon_lb once. if alpha=None, sweeps alpha from alpha_min to 10, runs minimize_scalar to find the tightest (lowest) epsilon

**`alpha_min = 1.0 / (1.0 + p_norm) + 1e-4`**
* lower bound on alpha from the paper, depends on which norm you're using for prediction error
* p_norm=2 (MSE) → alpha_min ≈ 0.334
* p_norm=1 (MAE) → alpha_min ≈ 0.5

---

## run_all.py

runs 5 experiments sequentially, logs everything (numbers + plain english interpretation) to `results/run_log.md`, saves a summary plot to `results/summary.png`. you run it once and come back to results you can actually read.

---

### global constants

**`N_REPS = 8`**
* how many times to repeat each MI estimate at each setting
* renyi_mi is random — every run shuffles data differently for the train/test split and gets a slightly different KDE. run it once and you get noise. run it 8 times and average to get closer to the true value.
* 8 is a compromise. more reps = more accurate but slower. could bump to 15-20 for a final run.
* ask: is 8 enough to get stable means, or are we still seeing a lot of run-to-run variance?

**`N_DATA = 600`**
* size of the synthetic datasets (Gaussian and log-normal experiments)
* NOT the Rayleigh dataset — that's always 1024 (fixed by how it's generated)
* 600 is enough that basic KDE should work but small enough that the run finishes fast
* ask: do our conclusions change at N=200 vs N=1000? worth running once to check

**`ALPHA_SWEEP = np.round(np.linspace(0.4, 4.0, 19), 2)`**
* 19 evenly spaced alpha values from 0.4 to 4.0
* starts at 0.4 not 0 — alpha=0 is undefined, and we already know very low alpha blows up
* stops at 4.0 — results are already getting sketchy at 3+, no point pushing further
* `np.round(..., 2)` just cleans up floating point so 1.0000000001 shows as 1.0 in the log

**`FRAC_SWEEP = np.round(np.linspace(0.2, 0.8, 13), 2)`**
* 13 train_frac values from 0.2 to 0.8
* 0.2 means 20% of data builds the KDE, 80% evaluates it
* don't go below 0.2 (too few points to build a decent KDE) or above 0.8 (too few held-out points to evaluate it, estimate gets noisy)

---

### logging functions

**`log_lines = []` and `def log(s='')`**
* every call to `log()` does two things: prints to terminal so you can watch it run, and appends to `log_lines` so it can be written to a file at the end
* `log()` with no argument just adds a blank line (for spacing)

**`def log_section(title)`**
* prints a markdown `##` header. makes the log file readable as a document.

**`def log_interpret(s)`**
* prints the line with a `>` prefix — markdown blockquote formatting
* visually separates the plain-english interpretation from the raw numbers in the log
* these are the lines that explain what the numbers actually mean

**`def save_log()`**
* joins all the logged lines with newlines and writes to `results/run_log.md`
* called once at the very end after all experiments finish

---

### interpretation helper functions

these are the functions that turn raw numbers into readable sentences. each one handles a specific type of result.

**`interpret_spread(spread, avg_std, context)`**
* spread = range of mean MI values across the sweep (max - min)
* avg_std = average noise level (how much MI varies run-to-run at the same setting)
* signal-to-noise = spread / avg_std
* if SNR < 1: variation is smaller than noise → you're just seeing randomness, not a real alpha effect
* if SNR 1-2.5: there's probably a real trend but it's weak, get more reps before concluding anything
* if SNR > 2.5: clear real signal, the sweep is showing you something true
* ask: is 2.5 the right threshold? what SNR do we need before we'd trust a result enough to tell Yi?

**`interpret_ksg_vs_kde(mi_ksg, mi_kde_at_1)`**
* compares the KSG estimate (ground truth for alpha=1) to what KDE gives at alpha=1
* they should be close since both are estimating Shannon MI on the same data
* a big gap means KDE is struggling on this dataset — fat tails, weird shape, etc.
* gap < 5%: KDE fine. 5-20%: acceptable. > 20%: worry about it.
* this is important because if KDE is way off at alpha=1 where we can check it, it's probably also wrong at other alpha values where we can't check

**`interpret_unreliable_alphas(unreliable, threshold_pct)`**
* takes the list of alpha values where noise > threshold_pct of mean
* explains WHY they're unreliable based on where the bad values are:
  * bad at low alpha: KDE amplifies rare events (low alpha formula), which are the hardest to estimate. expected.
  * bad at high alpha: raising density to a large power makes any KDE error explode. expected at extreme alpha.
  * bad across a wide range: the dataset itself is hard for KDE — probably fat-tailed or too small

**`interpret_direction(mi_low_alpha, mi_high_alpha, dataset_type)`**
* compares average MI at low vs high alpha (using only stable alpha values, not the blown-up ones)
* for log-normal (fat-tailed) data: expected to see MI decrease at high alpha, because high alpha ignores the tails where most of the interesting structure is
* if it goes the other way (MI increases at high alpha): unexpected, probably a KDE artifact
* if flat: the dependence is evenly spread across common and rare events

**`interpret_split(spread, low_std, high_std)`**
* checks two things: does the mean MI change across split ratios (it shouldn't much), and does noise change (it might at extreme ratios)
* low_std vs high_std comparison: checks if noise is worse at low train_frac (too little training data) or high train_frac (too little evaluation data)

**`interpret_rayleigh(eps_shannon, eps_kde_at_1, eps_best, best_alpha)`**
* three things to check on the Rayleigh sanity check:
  1. is the Shannon bound (KSG) actually tight? eps close to 0 means Pi* explains Y well. it should be — we're using the known correct Pi*.
  2. how far is KDE at alpha=1 from KSG? this is the "cost of generality" — we had to switch to KDE to support alpha, and KDE is worse.
  3. does the best alpha do better than Shannon? if yes, Renyi found something. if no, Shannon was already optimal or noise is drowning the signal.

---

### data generators

**`gaussian_pair(N, rho, seed)`**
* makes two correlated Gaussian variables with correlation coefficient rho
* `rng.multivariate_normal([0,0], cov, size=N)` — draws N samples from a 2D Gaussian
* cov = [[1, rho], [rho, 1]] means both variables have variance 1 and covariance rho
* rho=0.7 = pretty strong correlation (1.0 = perfectly correlated, 0 = independent)
* use this as the control: for Gaussian data, Renyi MI = Shannon MI for all alpha. so if the alpha sweep isn't flat here, something is wrong with the estimator.

**`lognormal_pair(N, rho, seed)`**
* generate a Gaussian pair, then `np.exp()` both variables
* exponentiating turns a Gaussian (bell curve) into a log-normal (right-skewed, fat tail)
* the correlation structure is the same as the Gaussian pair (rho=0.7) but the shape of the distribution is completely different
* "same copula, different marginals" — the way X and Y are linked is the same, but what X and Y individually look like is not
* this is where alpha should matter: the fat tail has rare/extreme events that alpha < 1 amplifies and alpha > 1 suppresses

**`independent_pair(N, seed)`**
* two completely independent Gaussians. MI should be ~0 for any alpha.
* used in tests to check that MI doesn't give false positives
* not used in the main experiments

---

### sweep helpers

**`sweep_alpha_reps(X, Y, alphas, n_reps, train_frac)`**
* outer loop: for each alpha in ALPHA_SWEEP
* inner loop: run renyi_mi n_reps times at that alpha
* returns arrays of means and stds, one entry per alpha value
* why repeat? renyi_mi shuffles data randomly every call — same inputs give slightly different outputs. n_reps runs lets you see: is the mean stable? is the std acceptable?
* NOT in original IT-PI. the original just runs the estimator once.

**`sweep_frac_reps(X, Y, fracs, alpha, n_reps)`**
* same structure but sweeping train_frac instead of alpha
* alpha is fixed at 1.5 — away from 1 so the bias from the split actually matters (at alpha=1 the bias roughly cancels, so exp 3 would be uninteresting)

**`summary_stats(means, stds, values, label)`**
* prints a subset of results as a markdown table (every ~8th row so it doesn't get huge)
* computes and logs three summary numbers:
  * spread = max(mean) - min(mean): did the thing we're sweeping actually change MI?
  * avg_std: how noisy is the estimator on average across the sweep?
  * SNR = spread / avg_std: is the variation real or just noise?
* returns spread and avg_std so the experiment functions can pass them to the interpret functions

---

### experiments

**`exp_gaussian_baseline()`**
* purpose: verify that the estimator is behaving correctly on easy data. if alpha changes MI a lot on Gaussian data, something is wrong.
* runs sweep_alpha_reps on gaussian_pair
* also runs KSG directly to get the "true" Shannon MI to compare against
* flags any alpha where std > 20% of mean as unreliable
* calls interpret_spread, interpret_ksg_vs_kde, interpret_unreliable_alphas to explain results in plain english
* ask: the first run showed alpha<1 blowing up even on Gaussian data. is that a KDE problem or is it expected behavior?

**`exp_nongaussian_baseline()`**
* purpose: show that alpha actually does something on non-Gaussian data, and see which direction
* runs sweep_alpha_reps on lognormal_pair
* only computes the direction check (low vs high alpha) on stable alpha values — important because including the blown-up low-alpha values would make the "low alpha average" meaningless
* `stable_mask = rel_std <= 0.2` filters to only the alpha values where the estimate is reliable before comparing directions
* ask: the first run showed the KSG vs KDE gap was 2839% on log-normal. that's massive. does it get better with more data?

**`exp_split_sweep()`**
* purpose: answer Yi's question — does the 50/50 split ratio actually matter?
* runs sweep_frac_reps on gaussian_pair at alpha=1.5
* checks both mean stability (does the split change the answer?) and variance pattern (is noise worse at extreme splits?)
* first run result: mean was flat (split doesn't change the answer at N=600), but noise was slightly higher at high train_frac which was unexpected
* ask: at what N does the split ratio start mattering? intuition says smaller N would be more sensitive but we haven't tested it

**`exp_alpha_reliability()`**
* purpose: find the actual cutoff — below what alpha are results too noisy to trust?
* runs 15 reps instead of 8 to get a more accurate noise estimate
* computes rel_std = std/mean for each alpha. > 0.25 = unreliable (25% noise)
* runs on BOTH Gaussian and log-normal so we can see if the cutoff depends on data type
* `bad = ALPHA_SWEEP[rel_std > 0.25]` — list of bad alpha values
* `stable = ALPHA_SWEEP[rel_std <= 0.25]` — the safe zone
* first run: Gaussian safe zone was alpha 1.0-4.0. log-normal safe zone was 1.2-4.0. so log-normal needs a higher minimum alpha.
* ask: should we add an automatic warning to renyi_mi itself that detects and flags when the estimate looks unstable?

**`exp_rayleigh()`**
* purpose: end-to-end sanity check on real physics data with a known correct answer
* uses Pi* = y/sqrt(mu*t) hardcoded in — skips CMA-ES entirely for speed
* coef = [[0.0], [1.0], [-0.5], [-0.5]] means: U^0 * y^1 * t^(-0.5) * mu^(-0.5) = y/sqrt(t*mu)
* computes eps via KSG (the reliable reference) then sweeps alpha with KDE
* the key comparison: eps_shannon (KSG, should be near 0 for correct Pi*) vs eps_kde_at_1 (KDE at same alpha=1, will be worse)
* first run: eps_shannon=0.079 (good), eps_kde_at_1=0.70 (bad). 792% gap. KDE is way off on Rayleigh data.
* ask: why is the KDE vs KSG gap so much bigger on Rayleigh than on Gaussian? what's different about this dataset?

---

### `make_plots(g_res, ng_res, split_res, rel_res, ray_res)`

* takes the result dicts from all 5 experiments and makes a 2x3 plot grid
* all 5 experiment results are passed in as arguments — this function only does plotting, no computation

**`_band(ax, x, m, s, color, label)`**
* inner helper function, only exists inside make_plots
* plots a line (mean) with a shaded band (mean ± std) around it
* the shaded band shows uncertainty — wide band = noisy estimator, narrow band = stable

**plot 1 (top left) — Gaussian MI vs alpha**
* should look like a roughly flat line with some wobble
* gray dashed line = KSG reference (the "right" answer at alpha=1)
* if the KDE line is far below the KSG line at alpha=1, the estimator has a problem

**plot 2 (top middle) — log-normal MI vs alpha**
* should slope downward from left to right (high alpha suppresses tails, lowering MI)
* blowup at low alpha will look like a spike off the top of the plot — that's the KDE breaking down

**plot 3 (top right) — split ratio sensitivity**
* should look flat across all train_frac values
* the shaded band tells you how noisy each split ratio is — wider at the extremes would make sense

**plot 4 (bottom left) — alpha reliability**
* y-axis = noise/signal (rel_std). red dashed line at 0.25 = the cutoff threshold
* any alpha where the curve is above the red line is unreliable
* two curves: blue = Gaussian, orange = log-normal. log-normal will be worse (higher curve) because KDE struggles more on fat-tailed data

**plot 5 (bottom middle) — Rayleigh eps vs alpha**
* y-axis = epsilon (lower = tighter bound = better)
* gray dashed line = KSG reference (the target). KDE curve should try to approach this.
* red dotted line = the alpha that gave the best (lowest) eps in the sweep

**plot 6 (bottom right) — reading guide**
* static text box explaining what to look for in each plot
* so the plot is self-contained and you don't need notes to read it

---

### `main` block

**`if __name__ == '__main__':`**
* only runs when you call `python run_all.py` directly, not if something imports run_all as a module
* logs the settings at the top of the file so the log is self-documenting
* calls all 5 experiment functions in order, stores their return dicts
* calls save_log() and make_plots() at the end
* total runtime printed at the end — useful for knowing how long to expect future runs to take

---

## big picture of how the files connect

```
buckingham_pi.py    — dimensional analysis, unchanged from original
      ↓
itpi.py             — CMA-ES runner
    uses kraskov_mi (KSG) internally for optimization
    uses renyi_mi   (KDE) for final bound
      ↓
renyi_mi.py         — renyi_entropy × 3 → MI → epsilon
      ↑
run_all.py          — calls renyi_mi directly (no CMA-ES) for sensitivity experiments
                    — calls itpi.run_itpi for Rayleigh (with CMA-ES)
```

the key split: CMA-ES always runs at alpha=1 with KSG (stable, accurate).
user's alpha only matters at the end when we compute the bound on Pi*.
everything in run_all.py is testing whether renyi_mi itself is trustworthy
before we commit to using it in real experiments.
