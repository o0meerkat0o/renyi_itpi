"""
run_all.py

Runs all experiments and logs results to results/run_log.md.
Each result comes with a plain-language interpretation so you can
read the log on its own without needing to re-analyze the numbers.

Usage:
    python run_all.py

Experiments:
    1. Gaussian baseline — MI across alpha for near-Gaussian data (should be flat)
    2. Non-Gaussian baseline — MI across alpha for skewed data (should vary)
    3. Train/test split sweep — does 50/50 actually matter?
    4. Alpha reliability — where does the estimate get too noisy to trust?
    5. Rayleigh full — known answer, compare Shannon vs best alpha bound
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from renyi_mi import renyi_mi
from itpi import kraskov_mi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, 'run_log.md')

N_REPS = 8       # repetitions per setting (for variance estimates)
N_DATA = 600     # dataset size for sensitivity experiments
ALPHA_SWEEP = np.round(np.linspace(0.4, 4.0, 19), 2)
FRAC_SWEEP  = np.round(np.linspace(0.2, 0.8, 13), 2)


# ---- logging ---------------------------------------------------------------

log_lines = []

def log(s=''):
    print(s)
    log_lines.append(s)

def log_section(title):
    log()
    log(f"## {title}")
    log()

def log_interpret(s):
    """Logs a plain-language interpretation line, visually distinct."""
    log(f"> {s}")

def save_log():
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nlog saved: {LOG_PATH}")


# ---- plain language helpers ------------------------------------------------

def interpret_spread(spread, avg_std, context='MI'):
    """
    Decides whether the variation seen across a sweep is real signal
    or just noise, and returns a plain-language verdict.
    """
    if avg_std == 0:
        return f"{context} was completely flat — no variation at all."
    snr = spread / avg_std
    if snr < 1.0:
        return (f"The spread in {context} ({spread:.4f}) is smaller than the "
                f"average noise ({avg_std:.4f}). The variation you see is probably "
                f"just randomness from the KDE, not a real effect of alpha.")
    elif snr < 2.5:
        return (f"There's a real trend in {context} (spread={spread:.4f}) but it's "
                f"only {snr:.1f}x the noise level — take it as suggestive, not definitive. "
                f"More reps or more data would help confirm it.")
    else:
        return (f"Clear real signal: {context} varies by {spread:.4f} across the sweep, "
                f"which is {snr:.1f}x the noise level. This is not just KDE randomness.")

def interpret_ksg_vs_kde(mi_ksg, mi_kde_at_1):
    """Compares KSG and KDE estimates at alpha=1 and explains the gap."""
    gap = abs(mi_ksg - mi_kde_at_1)
    pct = 100 * gap / (mi_ksg + 1e-9)
    if pct < 5:
        verdict = "basically identical — KDE is working well here."
    elif pct < 20:
        verdict = (f"a {pct:.0f}% gap. KSG is more accurate at alpha=1, "
                   f"so KDE is slightly off but in the right ballpark.")
    else:
        verdict = (f"a {pct:.0f}% gap — that's large. KDE is struggling on "
                   f"this dataset at alpha=1. Be cautious about trusting KDE results here.")
    return f"KSG={mi_ksg:.4f} vs KDE={mi_kde_at_1:.4f}: {verdict}"

def interpret_unreliable_alphas(unreliable, threshold_pct):
    """Explains which alpha values are too noisy to use."""
    if len(unreliable) == 0:
        return f"All alpha values were stable (noise < {threshold_pct}% of mean). Safe to use the full sweep."
    lo, hi = unreliable[0], unreliable[-1]
    if hi < 1.0:
        return (f"Alpha values below {hi:.1f} are unreliable — noise is more than "
                f"{threshold_pct}% of the estimate. Stick to alpha >= {hi + 0.2:.1f}. "
                f"This is the KDE breaking down: low alpha amplifies rare events which "
                f"are the hardest to estimate accurately.")
    elif lo > 1.0:
        return (f"Alpha values above {lo:.1f} are unreliable. High alpha over-weights "
                f"common events and raises density to large powers, amplifying KDE errors.")
    else:
        return (f"A wide range of alpha is unreliable: {lo:.1f} to {hi:.1f}. "
                f"This suggests the dataset itself is hard to estimate density on — "
                f"possibly fat-tailed or too small for KDE to work well.")

def interpret_direction(mi_low_alpha, mi_high_alpha, dataset_type):
    """Interprets whether MI goes up or down with alpha and why."""
    if mi_high_alpha < mi_low_alpha * 0.95:
        return (f"MI decreases as alpha increases ({mi_low_alpha:.4f} → {mi_high_alpha:.4f}). "
                f"This is expected for {dataset_type} data: high alpha focuses on the dense "
                f"central region and ignores the tails. If the relationship between X and Y "
                f"is mostly carried by rare/extreme values, high alpha will undercount it.")
    elif mi_high_alpha > mi_low_alpha * 1.05:
        return (f"MI increases as alpha increases ({mi_low_alpha:.4f} → {mi_high_alpha:.4f}). "
                f"This is unexpected — higher alpha should suppress rare events. "
                f"Could be a KDE artifact or the dataset has an unusual dependence structure.")
    else:
        return (f"MI stays roughly flat across alpha ({mi_low_alpha:.4f} → {mi_high_alpha:.4f}). "
                f"For {dataset_type} data this means alpha doesn't matter much — "
                f"the dependence structure is spread evenly across common and rare events.")

def interpret_split(spread, low_std, high_std):
    """Interprets the train/test split sweep results."""
    if spread < 0.05:
        mean_line = "The MI estimate barely changes across split ratios — 50/50 is fine, but so is anything from 30/70 to 70/30."
    else:
        mean_line = f"The MI estimate shifts by {spread:.4f} across split ratios — the split ratio actually matters here."

    if low_std > high_std * 1.5:
        noise_line = ("Noise is higher at low train_frac (few training points = rough KDE = noisy estimates). "
                      "Don't go below 0.3.")
    elif high_std > low_std * 1.5:
        noise_line = ("Noise is surprisingly higher at high train_frac (few evaluation points = noisy average). "
                      "This is unusual — could just be randomness at this N.")
    else:
        noise_line = "Noise level is consistent across split ratios at this N. The split ratio choice doesn't matter much."

    return mean_line + " " + noise_line

def interpret_rayleigh(eps_shannon, eps_kde_at_1, eps_best, best_alpha):
    """Interprets Rayleigh results in terms of what the bound actually means."""
    lines = []

    # Shannon bound interpretation
    if eps_shannon < 0.15:
        lines.append(f"Shannon bound (KSG): eps={eps_shannon:.4f}. Very tight — Pi* explains almost all of Y. "
                     f"This is what we expect for the correct dimensionless group.")
    elif eps_shannon < 0.4:
        lines.append(f"Shannon bound (KSG): eps={eps_shannon:.4f}. Moderate — Pi* explains most of Y but not all.")
    else:
        lines.append(f"Shannon bound (KSG): eps={eps_shannon:.4f}. Loose — Pi* isn't explaining Y very well. "
                     f"This would be a problem if we didn't already know Pi* is correct.")

    # KDE vs KSG gap
    gap_pct = 100 * abs(eps_kde_at_1 - eps_shannon) / (eps_shannon + 1e-9)
    if gap_pct > 30:
        lines.append(f"KDE at alpha=1 gives eps={eps_kde_at_1:.4f} vs KSG eps={eps_shannon:.4f} — "
                     f"a {gap_pct:.0f}% gap. KDE is much less accurate than KSG on this dataset. "
                     f"This is the cost of switching to KDE to support general alpha.")
    else:
        lines.append(f"KDE at alpha=1 gives eps={eps_kde_at_1:.4f}, close to KSG eps={eps_shannon:.4f} "
                     f"({gap_pct:.0f}% gap). KDE is working reasonably well here.")

    # Best alpha
    if eps_best < eps_shannon:
        lines.append(f"Best alpha={best_alpha:.2f} gives eps={eps_best:.4f}, tighter than Shannon. "
                     f"Renyi is finding something Shannon missed — the relationship has structure "
                     f"that alpha={best_alpha:.2f} is better at capturing.")
    else:
        lines.append(f"Best alpha={best_alpha:.2f} gives eps={eps_best:.4f}, not better than Shannon. "
                     f"On this dataset, tuning alpha doesn't help — Shannon is already optimal or KDE noise is dominating.")

    return " ".join(lines)


# ---- data generators -------------------------------------------------------

def gaussian_pair(N=N_DATA, rho=0.7, seed=0):
    """Bivariate Gaussian, correlation rho. Renyi MI should be ~flat across alpha."""
    rng = np.random.default_rng(seed)
    cov = [[1, rho], [rho, 1]]
    data = rng.multivariate_normal([0, 0], cov, size=N)
    return data[:, 0], data[:, 1]

def lognormal_pair(N=N_DATA, rho=0.7, seed=0):
    """
    Log-normal marginals with Gaussian copula.
    Skewed / fat-tailed — Renyi MI should vary with alpha.
    alpha > 1 should weight the dense region more (lower MI).
    alpha < 1 should amplify the tail events (higher MI? or noisier).
    """
    rng = np.random.default_rng(seed)
    cov = [[1, rho], [rho, 1]]
    Z = rng.multivariate_normal([0, 0], cov, size=N)
    return np.exp(Z[:, 0]), np.exp(Z[:, 1])

def independent_pair(N=N_DATA, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=N), rng.normal(size=N)


# ---- experiment helpers ----------------------------------------------------

def sweep_alpha_reps(X, Y, alphas=ALPHA_SWEEP, n_reps=N_REPS, train_frac=0.5):
    """Estimate MI at each alpha, repeated n_reps times. Returns means and stds."""
    means, stds = [], []
    for a in alphas:
        vals = [renyi_mi(X, Y, alpha=float(a), train_frac=train_frac)
                for _ in range(n_reps)]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)

def sweep_frac_reps(X, Y, fracs=FRAC_SWEEP, alpha=1.5, n_reps=N_REPS):
    """Estimate MI at each train_frac, repeated n_reps times."""
    means, stds = [], []
    for frac in fracs:
        vals = [renyi_mi(X, Y, alpha=alpha, train_frac=float(frac))
                for _ in range(n_reps)]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)

def summary_stats(means, stds, values, label):
    """Log a compact summary table."""
    log(f"| {label} | mean MI | std |")
    log("|---|---|---|")
    step = max(1, len(values) // 8)
    for i in range(0, len(values), step):
        log(f"| {values[i]:.2f} | {means[i]:.4f} | {stds[i]:.4f} |")
    log()
    spread = means.max() - means.min()
    mean_std = stds.mean()
    log(f"- range of means: {means.min():.4f} – {means.max():.4f}  (spread={spread:.4f})")
    log(f"- avg std across reps: {mean_std:.4f}")
    log(f"- signal-to-noise (spread / avg_std): {spread/mean_std:.2f}" if mean_std > 0 else "")
    return spread, mean_std


# ---- experiment 1: Gaussian baseline ---------------------------------------

def exp_gaussian_baseline():
    log_section("Experiment 1: Gaussian baseline (alpha sweep)")
    log("Dataset: bivariate Gaussian, rho=0.7, N=600")
    log("Question: does MI vary with alpha? For Gaussian data it shouldn't much.")
    log()

    X, Y = gaussian_pair()
    t0 = time.time()
    means, stds = sweep_alpha_reps(X, Y)
    elapsed = time.time() - t0

    spread, avg_std = summary_stats(means, stds, ALPHA_SWEEP, "alpha")
    log(f"- time: {elapsed:.1f}s")

    mi_ksg = kraskov_mi(X.reshape(-1,1), Y.reshape(-1,1))
    mi_kde_at_1 = means[np.argmin(np.abs(ALPHA_SWEEP - 1.0))]
    log(f"- KSG (Shannon) reference: {mi_ksg:.4f}")
    log(f"- KDE at alpha=1.0: {mi_kde_at_1:.4f}")

    rel_std = stds / (means + 1e-9)
    unreliable = ALPHA_SWEEP[rel_std > 0.2]
    if len(unreliable):
        log(f"- unreliable alpha (std > 20% of mean): {unreliable.tolist()}")
    else:
        log("- all alpha values appear stable (std < 20% of mean)")

    log()
    log("**what this means:**")
    log_interpret(interpret_spread(spread, avg_std, "MI"))
    log_interpret(interpret_ksg_vs_kde(mi_ksg, mi_kde_at_1))
    log_interpret(interpret_unreliable_alphas(unreliable, 20))

    return {'alphas': ALPHA_SWEEP, 'means': means, 'stds': stds, 'ksg': mi_ksg}


# ---- experiment 2: non-Gaussian baseline -----------------------------------

def exp_nongaussian_baseline():
    log_section("Experiment 2: Non-Gaussian baseline (alpha sweep)")
    log("Dataset: log-normal marginals with Gaussian copula, rho=0.7, N=600")
    log("Question: does MI vary with alpha? For non-Gaussian data it should.")
    log("Expected: alpha>1 suppresses tails → lower MI. alpha<1 amplifies tails → higher or noisier.")
    log()

    X, Y = lognormal_pair()
    t0 = time.time()
    means, stds = sweep_alpha_reps(X, Y)
    elapsed = time.time() - t0

    spread, avg_std = summary_stats(means, stds, ALPHA_SWEEP, "alpha")
    log(f"- time: {elapsed:.1f}s")

    mi_ksg = kraskov_mi(X.reshape(-1,1), Y.reshape(-1,1))
    mi_kde_at_1 = means[np.argmin(np.abs(ALPHA_SWEEP - 1.0))]
    log(f"- KSG reference: {mi_ksg:.4f}")
    log(f"- KDE at alpha=1.0: {mi_kde_at_1:.4f}")

    rel_std = stds / (means + 1e-9)
    unreliable = ALPHA_SWEEP[rel_std > 0.2]
    if len(unreliable):
        log(f"- unreliable alpha: {unreliable.tolist()}")
    else:
        log("- all alpha values stable")

    # only compute direction on stable alpha values
    stable_mask = rel_std <= 0.2
    stable_alphas = ALPHA_SWEEP[stable_mask]
    stable_means  = means[stable_mask]
    if len(stable_means) > 0:
        mi_low  = stable_means[stable_alphas <= 1.2].mean() if any(stable_alphas <= 1.2) else float('nan')
        mi_high = stable_means[stable_alphas >= 2.5].mean() if any(stable_alphas >= 2.5) else float('nan')
    else:
        mi_low, mi_high = float('nan'), float('nan')
    log(f"- avg MI at low alpha (stable only): {mi_low:.4f}" if not np.isnan(mi_low) else "- no stable low-alpha values")
    log(f"- avg MI at high alpha (stable only): {mi_high:.4f}" if not np.isnan(mi_high) else "- no stable high-alpha values")

    log()
    log("**what this means:**")
    log_interpret(interpret_spread(spread, avg_std, "MI"))
    log_interpret(interpret_ksg_vs_kde(mi_ksg, mi_kde_at_1))
    log_interpret(interpret_unreliable_alphas(unreliable, 20))
    if not np.isnan(mi_low) and not np.isnan(mi_high):
        log_interpret(interpret_direction(mi_low, mi_high, "log-normal"))
    else:
        log_interpret("Not enough stable alpha values to determine direction reliably.")

    return {'alphas': ALPHA_SWEEP, 'means': means, 'stds': stds, 'ksg': mi_ksg}


# ---- experiment 3: train/test split sweep ----------------------------------

def exp_split_sweep():
    log_section("Experiment 3: Train/test split sensitivity")
    log("Dataset: bivariate Gaussian, rho=0.7, N=600")
    log("Question: does the 50/50 split ratio actually matter?")
    log("Expected: at N=600, mean stays flat, std might rise at very low train_frac.")
    log()

    X, Y = gaussian_pair()
    t0 = time.time()
    means, stds = sweep_frac_reps(X, Y, alpha=1.5)
    elapsed = time.time() - t0

    spread, avg_std = summary_stats(means, stds, FRAC_SWEEP, "train_frac")
    log(f"- time: {elapsed:.1f}s")

    low_std  = stds[FRAC_SWEEP <= 0.3].mean()
    high_std = stds[FRAC_SWEEP >= 0.6].mean()
    log(f"- avg std at train_frac<=0.3: {low_std:.4f}")
    log(f"- avg std at train_frac>=0.6: {high_std:.4f}")

    log()
    log("**what this means:**")
    log_interpret(interpret_split(spread, low_std, high_std))

    return {'fracs': FRAC_SWEEP, 'means': means, 'stds': stds}


# ---- experiment 4: alpha reliability cutoff --------------------------------

def exp_alpha_reliability():
    log_section("Experiment 4: Alpha reliability cutoff")
    log("Run the same (X,Y) pair 15 times at each alpha.")
    log("Where does variance blow up? That's the unreliable zone.")
    log("Testing both Gaussian and log-normal so we can compare.")
    log()

    results = {}
    for name, (X, Y) in [('gaussian', gaussian_pair()), ('lognormal', lognormal_pair())]:
        log(f"### {name}")
        means, stds = sweep_alpha_reps(X, Y, n_reps=15)
        rel_std = stds / (means + 1e-9)

        bad = ALPHA_SWEEP[rel_std > 0.25]
        cutoff = bad[0] if len(bad) else None
        log(f"- rel_std > 0.25 at alpha: {bad.tolist()}")
        log(f"- suggested cutoff: {'alpha > ' + str(cutoff) if cutoff else 'none found in sweep'}")

        # find where things stabilize
        stable = ALPHA_SWEEP[rel_std <= 0.25]
        if len(stable):
            log(f"- stable range: alpha {stable[0]:.2f} to {stable[-1]:.2f}")

        log()
        log("**what this means:**")
        log_interpret(interpret_unreliable_alphas(bad, 25))
        if len(stable):
            pct_stable = 100 * len(stable) / len(ALPHA_SWEEP)
            log_interpret(f"{pct_stable:.0f}% of the alpha sweep range is usable for {name} data. "
                          f"Safe zone: alpha {stable[0]:.1f} – {stable[-1]:.1f}.")
        log()

        results[name] = {'alphas': ALPHA_SWEEP, 'means': means, 'stds': stds, 'rel_std': rel_std}

    return results


# ---- experiment 5: Rayleigh ------------------------------------------------

def exp_rayleigh():
    log_section("Experiment 5: Rayleigh — known answer")
    log("Known Pi*: y / sqrt(mu * t). Known bound should approach 0.")
    log("Running Shannon (KSG) vs alpha sweep on the converged Pi*.")
    log("This is the sanity check — if the bound behaves here, we trust it elsewhere.")
    log()

    try:
        from scipy.special import erf
        from buckingham_pi import calc_pi_omega

        np.random.seed(42)
        rows = []
        for U  in np.random.uniform(0.5, 1.0, 4):
            for mu in np.random.uniform(1e-3, 1e-2, 4):
                for y  in np.linspace(0.02, 0.9, 8):
                    for t  in np.linspace(4, 10, 8):
                        u = U * (1 - erf(y / (2 * np.sqrt(mu * t))))
                        rows.append([U, y, t, mu, u])
        data = np.array(rows)
        X_ray = data[:, :4]
        Y_ray = data[:, 4] / data[:, 0]

        # use known correct Pi* directly (skip CMA-ES for speed)
        # Pi* = y / sqrt(mu * t) = y^1 * t^-0.5 * mu^-0.5
        # variable order: U, y, t, mu
        coef = np.array([[0.0], [1.0], [-0.5], [-0.5]])
        pi_star = np.ones((X_ray.shape[0], 1))
        for i in range(X_ray.shape[1]):
            pi_star *= (X_ray[:, i] ** coef[i, 0]).reshape(-1, 1)

        Y_col = Y_ray.reshape(-1, 1)

        mi_ksg = kraskov_mi(pi_star, Y_col)
        eps_shannon = float(np.exp(-max(mi_ksg, 0)))
        log(f"- Shannon MI (KSG): {mi_ksg:.4f}  →  eps={eps_shannon:.4f}")

        means, stds = sweep_alpha_reps(pi_star.flatten(), Y_ray, n_reps=6)
        eps_curve = np.exp(-np.clip(means, 0, None))
        best_eps_idx = np.argmin(eps_curve)
        best_alpha = ALPHA_SWEEP[best_eps_idx]
        best_eps = eps_curve[best_eps_idx]

        eps_kde_at_1 = eps_curve[np.argmin(np.abs(ALPHA_SWEEP - 1.0))]
        eps_at_2 = eps_curve[np.argmin(np.abs(ALPHA_SWEEP - 2.0))]

        log(f"- KDE eps at alpha=1.0: {eps_kde_at_1:.4f}")
        log(f"- KDE eps at alpha=2.0: {eps_at_2:.4f}")
        log(f"- best alpha: {best_alpha:.2f}  →  eps={best_eps:.4f}")

        log()
        log("**what this means:**")
        log_interpret(interpret_rayleigh(eps_shannon, eps_kde_at_1, best_eps, best_alpha))

        return {'alphas': ALPHA_SWEEP, 'eps': eps_curve, 'stds': stds,
                'eps_shannon': eps_shannon, 'best_alpha': best_alpha, 'best_eps': best_eps}

    except Exception as e:
        log(f"Rayleigh experiment failed: {e}")
        return {}


# ---- plots -----------------------------------------------------------------

def make_plots(g_res, ng_res, split_res, rel_res, ray_res):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Renyi MI Experiments — Summary', fontsize=13)

    def _band(ax, x, m, s, color, label):
        ax.plot(x, m, color=color, lw=2, label=label)
        ax.fill_between(x, m-s, m+s, alpha=0.2, color=color)

    # 1: Gaussian alpha sweep
    _band(axes[0,0], g_res['alphas'], g_res['means'], g_res['stds'], 'steelblue', 'KDE mean±std')
    axes[0,0].axhline(g_res['ksg'], color='gray', ls='--', label=f"KSG ref={g_res['ksg']:.3f}")
    axes[0,0].set(title='Exp 1: Gaussian MI vs alpha\n(expect ~flat line)', xlabel='alpha', ylabel='MI')
    axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)

    # 2: non-Gaussian alpha sweep
    _band(axes[0,1], ng_res['alphas'], ng_res['means'], ng_res['stds'], 'darkorange', 'KDE mean±std')
    axes[0,1].axhline(ng_res['ksg'], color='gray', ls='--', label=f"KSG ref={ng_res['ksg']:.3f}")
    axes[0,1].set(title='Exp 2: Log-normal MI vs alpha\n(expect downward slope)', xlabel='alpha', ylabel='MI')
    axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.3)

    # 3: split sweep
    _band(axes[0,2], split_res['fracs'], split_res['means'], split_res['stds'], 'seagreen', 'mean±std')
    axes[0,2].axvline(0.5, color='gray', ls='--', label='default 0.5')
    axes[0,2].set(title='Exp 3: Split ratio sensitivity\n(expect flat — ratio should not matter)', xlabel='train_frac', ylabel='MI')
    axes[0,2].legend(fontsize=8); axes[0,2].grid(alpha=0.3)

    # 4: reliability — relative std
    for name, color in [('gaussian','steelblue'), ('lognormal','darkorange')]:
        if name in rel_res:
            r = rel_res[name]
            axes[1,0].plot(r['alphas'], r['rel_std'], color=color, lw=2, label=name)
    axes[1,0].axhline(0.25, color='red', ls='--', label='25% noise threshold (cutoff)')
    axes[1,0].set(title='Exp 4: Alpha reliability\n(above red line = too noisy to trust)', xlabel='alpha', ylabel='noise / signal')
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

    # 5: Rayleigh eps vs alpha
    if ray_res and 'eps' in ray_res:
        axes[1,1].plot(ray_res['alphas'], ray_res['eps'], 'purple', lw=2, label='KDE eps (lower = better)')
        axes[1,1].axhline(ray_res['eps_shannon'], color='gray', ls='--',
                          label=f"KSG target={ray_res['eps_shannon']:.3f}")
        if 'best_alpha' in ray_res:
            axes[1,1].axvline(ray_res['best_alpha'], color='red', ls=':',
                              label=f"best alpha={ray_res['best_alpha']:.2f}")
        axes[1,1].set(title='Exp 5: Rayleigh eps vs alpha\n(lower = tighter bound, KSG is the target)', xlabel='alpha', ylabel='epsilon_lb')
        axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Rayleigh skipped', ha='center', va='center', transform=axes[1,1].transAxes)

    # 6: reading guide
    axes[1,2].axis('off')
    guide = (
        "HOW TO READ THESE PLOTS\n\n"
        "Exp 1 (Gaussian): flat line = good.\n"
        "Any slope = KDE artifact, not real.\n\n"
        "Exp 2 (Log-normal): downward slope\n"
        "expected. Blowup at low alpha = KDE\n"
        "struggling on fat-tailed data.\n\n"
        "Exp 3 (Split): flat = split ratio\n"
        "doesn't matter at this N.\n\n"
        "Exp 4 (Reliability): stay below\n"
        "the red line. That's your safe zone.\n\n"
        "Exp 5 (Rayleigh): purple curve\n"
        "should approach the gray dashed line.\n"
        "Gap = cost of using KDE over KSG."
    )
    axes[1,2].text(0.05, 0.97, guide, va='top', fontsize=9, family='monospace',
                   transform=axes[1,2].transAxes)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"plot saved: {out}")


# ---- main ------------------------------------------------------------------

if __name__ == '__main__':
    start = time.time()

    log("# Renyi MI Experiment Log")
    log()
    log("Results include plain-language interpretation after each number block.")
    log("Lines starting with '>' are the interpretations.")
    log()
    log(f"- N_DATA={N_DATA}, N_REPS={N_REPS}")
    log(f"- alpha sweep: {ALPHA_SWEEP[0]} to {ALPHA_SWEEP[-1]}, {len(ALPHA_SWEEP)} values")
    log(f"- train_frac sweep: {FRAC_SWEEP[0]} to {FRAC_SWEEP[-1]}, {len(FRAC_SWEEP)} values")
    log()
    log("---")

    g_res     = exp_gaussian_baseline()
    ng_res    = exp_nongaussian_baseline()
    split_res = exp_split_sweep()
    rel_res   = exp_alpha_reliability()
    ray_res   = exp_rayleigh()

    total = time.time() - start
    log()
    log("---")
    log(f"## Total runtime: {total/60:.1f} min")

    save_log()
    make_plots(g_res, ng_res, split_res, rel_res, ray_res)
    print("\nDone. Results are in results/run_log.md — readable on their own.")
