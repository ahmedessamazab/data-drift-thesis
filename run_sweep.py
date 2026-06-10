"""
Parameter Sweep Driver for the IPNN Drift-Detection Framework
=============================================================
Runs the detection pipeline (run_full_experiment.run_experiment) over a FULL
GRID of hyper-parameters and MULTIPLE RANDOM SEEDS, then writes:

  sweep_results.csv       one row per (parameter-combination x seed) run
  sweep_aggregated.csv    seed-averaged results (mean +/- std) per combination
  sweep_fig1_detection_vs_Q.png
  sweep_fig2_delay_vs_Q.png
  sweep_fig3_alarms_vs_tau.png
  sweep_fig4_detection_vs_warmup.png
  sweep_fig5_heatmap_Q_tau.png
  sweep_fig6_seed_variability.png

WHY MULTIPLE SEEDS
------------------
Detection delays for the near-threshold variance drifts are sensitive to the
exact noise realisation.  Averaging several seeds and reporting mean +/- std
gives robust, defensible numbers instead of a single brittle run.

USAGE
-----
    python run_sweep.py                 # full default grid x 5 seeds (asks to confirm)
    python run_sweep.py --quick         # tiny grid x 2 seeds, for a fast sanity check
    python run_sweep.py --yes           # skip the confirmation prompt
    python run_sweep.py --seeds 42 7 1  # custom seeds
    python run_sweep.py --out my_sweep  # custom output folder

EDIT THE GRID below (the GRID dict and SEEDS list) to add/remove parameter
values.  Parameters not listed in GRID are held at the thesis primary value.
"""

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Keep each worker process single-threaded for BLAS so that running many
# processes in parallel does not oversubscribe the CPU cores. Must be set
# before numpy is imported (children inherit these on spawn).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_full_experiment import run_experiment, DEFAULT_CFG

# ──────────────────────────────────────────────────────────────────────────────
# EDIT ME — the hyper-parameter grid and the random seeds
# ──────────────────────────────────────────────────────────────────────────────
GRID = {
    "Q":                  [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6],  # series-growth exponent (9)
    "tau":                [0.06, 0.07, 0.08, 0.085, 0.09, 0.10, 0.12],       # ISE threshold          (7, fine)
    "warmup":             [200, 250,300, 400],                                        # warm-up w  (add 200, 250 for more)
    "k":                  [2, 4, 3.0, 6.0],                                        # series scale (add 2.0, 4.0 for more)
    "gamma":              [0.6, 0.7, 0.8, 1.0],                              # learning-rate exponent (4, fine)
    "confirmation_steps": [5, 7, 10],                                        # confirmation window c  (3)
}
SEEDS = [42, 7, 123, 2024, 99]                                              # (5 seeds)
# Default size: 9*7*2*2*4*3 = 3024 combinations x 5 seeds = 15120 runs (~2 h).
# tau and gamma get the finest resolution (they are the strongest levers for the
# detection / false-alarm trade-off). To go bigger, widen the warmup or k lists.
# The (machine-calibrated) time estimate is printed before the run starts.

# A small grid used by --quick (full N is still required, so this is ~8 runs).
QUICK_GRID = {"Q": [0.2, 0.5], "tau": [0.08, 0.10], "warmup": [300], "k": [3.0], "gamma": [1.0]}
QUICK_SEEDS = [42, 7]

METRIC_COLS = ["N", "seed", "Q", "k", "gamma", "warmup", "tau", "confirmation_steps",
               "n_drifts", "n_detected", "tpr", "total_alarms", "matched_alarms",
               "unmatched_alarms", "unmatched_ratio", "delay_min", "delay_max",
               "delay_mean", "delay_median", "max_ise", "mean_background_ise",
               "threshold_exceedances", "delaytype_mean", "delaytype_variance",
               "delaytype_gradual", "delaytype_cyclic", "delaytype_distribution"]


def expand_grid(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


# The dominant cost of a run is sum_n (M * q(n)) ~ M * k * N^(Q+1)/(Q+1).
# Constants below are calibrated to a real run on this project's hardware: the
# previous 2520-run sweep finished in ~20 min, i.e. a k=3, Q=0.5, N=20000 run
# takes ~0.5 s (≈0.15 s fixed overhead + ~0.35 s density work). Adjust the two
# coefficients if your machine is noticeably faster or slower.
_REF = 3.0 * (20000 ** 1.5) / 1.5    # k=3, Q=0.5, N=20000


def estimate_seconds(combo, N):
    Q = combo.get("Q", DEFAULT_CFG["Q"])
    k = combo.get("k", DEFAULT_CFG["k"])
    work = k * (N ** (Q + 1.0)) / (Q + 1.0)
    return 0.15 * (N / 20000.0) + 0.35 * work / _REF


_KEY_FIELDS = ("Q", "tau", "warmup", "k", "gamma", "confirmation_steps", "seed")


def _task_key(combo, seed):
    """A hashable identity for a (combination, seed) run, used to skip duplicates."""
    return tuple(round(float(combo.get(f, DEFAULT_CFG.get(f))), 6) if f != "seed" else int(seed)
                 for f in _KEY_FIELDS)


def _run_task(task):
    """Worker: run one experiment and return a compact result row (no big arrays)."""
    combo, seed, N = task
    cfg = {**combo, "seed": seed, "N": N}
    m = run_experiment(cfg, make_plots=False, save_csvs=False, verbose=False)
    row = {c: m.get(c) for c in dict.fromkeys(METRIC_COLS)}
    row["alarm_positions"] = ";".join(str(a) for a in m.get("alarm_positions", []))
    # tiny progress payload for the parent to print
    row["_det"], row["_drifts"], row["_alarms"] = m["n_detected"], m["n_drifts"], m["total_alarms"]
    return row


def _load_done(path):
    """Return (existing_rows, done_key_set) from a previous sweep_results.csv, if any."""
    if not path or not os.path.exists(path):
        return [], set()
    prev = pd.read_csv(path)
    done = set()
    for _, r in prev.iterrows():
        done.add(tuple(round(float(r[f]), 6) if f != "seed" else int(r["seed"]) for f in _KEY_FIELDS))
    cols = [c for c in prev.columns if not c.startswith("_")]
    return prev[cols].to_dict("records"), done


def run_sweep(grid, seeds, out_dir, n_override=None, confirm=True, jobs=1, resume=None):
    combos = expand_grid(grid)
    N = n_override or DEFAULT_CFG["N"]

    # All (combo, seed) tasks; optionally skip ones already present (resume).
    rows, done_keys = _load_done(resume)
    if rows:
        print(f"  Resuming: found {len(rows)} completed runs in {resume} (these will be skipped)")
    tasks = [(c, s, N) for c in combos for s in seeds if _task_key(c, s) not in done_keys]
    n_total = len(combos) * len(seeds)
    n_todo = len(tasks)
    est_total = sum(estimate_seconds(c, N) for c, s, _ in tasks)
    jobs = max(1, int(jobs))
    eff_jobs = min(jobs, os.cpu_count() or 1)

    print("=" * 70)
    print(f"  Parameter grid : { {k: grid[k] for k in grid} }")
    print(f"  Seeds          : {seeds}")
    print(f"  Combinations   : {len(combos)} x {len(seeds)} seeds = {n_total} runs"
          + (f"  ({n_todo} still to run)" if n_todo != n_total else ""))
    print(f"  Stream length  : N = {N}")
    print(f"  Parallel jobs  : {eff_jobs} of {os.cpu_count()} CPU cores")
    print(f"  Estimated time : ~{est_total/eff_jobs/3600:.1f} h ({est_total/eff_jobs/60:.0f} min) "
          f"on {eff_jobs} cores  [sequential would be ~{est_total/3600:.1f} h]")
    print("=" * 70)
    if confirm:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return None

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "sweep_results.csv")
    t0 = time.time()
    save_cols = list(dict.fromkeys(METRIC_COLS)) + ["alarm_positions"]

    def save():
        pd.DataFrame([{k: r.get(k) for k in save_cols} for r in rows]).to_csv(out_csv, index=False)

    def progress(row, done):
        elapsed = time.time() - t0
        eta = (elapsed / done) * (n_todo - done)
        print(f"  [{done:>5}/{n_todo}] Q={row['Q']} tau={row['tau']} w={row['warmup']} k={row['k']} "
              f"g={row['gamma']} c={row['confirmation_steps']} seed={row['seed']} -> "
              f"det {row['_det']}/{row['_drifts']}, alarms {row['_alarms']} | ETA {eta/60:.1f} min")

    if eff_jobs == 1:
        for done, task in enumerate(tasks, 1):
            row = _run_task(task)
            rows.append(row)
            progress(row, done)
            if done % 25 == 0:
                save()
    else:
        with ProcessPoolExecutor(max_workers=eff_jobs) as ex:
            futs = {ex.submit(_run_task, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done += 1
                progress(row, done)
                if done % 50 == 0:
                    save()
    save()

    df = pd.DataFrame([{k: r.get(k) for k in save_cols} for r in rows])
    print(f"\n  Wrote {out_csv}  ({len(df)} rows)")

    agg = aggregate(df)
    agg.to_csv(os.path.join(out_dir, "sweep_aggregated.csv"), index=False)
    print(f"  Wrote {os.path.join(out_dir, 'sweep_aggregated.csv')}  ({len(agg)} combinations)")

    best = rank_configs(agg)
    best.to_csv(os.path.join(out_dir, "sweep_best_configs.csv"), index=False)
    print(f"  Wrote {os.path.join(out_dir, 'sweep_best_configs.csv')}  "
          f"(ranked: most drifts detected, then fewest false alarms)")
    _print_top(best)

    make_sweep_figures(df, out_dir)
    print(f"\n  All sweep outputs are in: {out_dir}/")
    return df


def rank_configs(agg):
    """
    Rank seed-averaged configurations to find the best detector:
    first by most drifts detected, then by fewest false alarms (unmatched),
    then by fastest (smallest median) detection delay.

    The top row is the configuration that detects the most drifts with the
    lowest false-alarm count — i.e. the answer to "which parameters detect the
    drift with very low false alarms".
    """
    a = agg.copy()
    sort_cols, asc = [], []
    if "n_detected_mean" in a:        sort_cols.append("n_detected_mean");      asc.append(False)
    if "unmatched_alarms_mean" in a:  sort_cols.append("unmatched_alarms_mean"); asc.append(True)
    if "delay_median_mean" in a:      sort_cols.append("delay_median_mean");     asc.append(True)
    return a.sort_values(by=sort_cols, ascending=asc).reset_index(drop=True)


def _print_top(best, n=8):
    cols = [c for c in ["Q", "k", "gamma", "warmup", "tau", "confirmation_steps", "n_detected_mean",
                        "unmatched_alarms_mean", "total_alarms_mean", "delay_median_mean"]
            if c in best.columns]
    show = best[cols].head(n).copy()
    rename = {"n_detected_mean": "detected", "unmatched_alarms_mean": "false_alarms",
              "total_alarms_mean": "alarms", "delay_median_mean": "median_delay"}
    show = show.rename(columns=rename)
    print("\n  Top configurations (seed-averaged) — most detected, fewest false alarms:")
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def aggregate(df):
    group_keys = ["Q", "k", "gamma", "warmup", "tau", "confirmation_steps", "N"]
    metrics = ["n_detected", "tpr", "total_alarms", "unmatched_alarms",
               "delay_mean", "delay_median", "max_ise", "mean_background_ise"]
    g = df.groupby(group_keys, dropna=False)
    out = g[metrics].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def _slice_holding_defaults(df, vary):
    """Rows where every swept parameter except `vary` equals its primary value."""
    hold = {"Q": DEFAULT_CFG["Q"], "tau": DEFAULT_CFG["tau"], "warmup": DEFAULT_CFG["warmup"],
            "k": DEFAULT_CFG["k"], "gamma": DEFAULT_CFG["gamma"],
            "confirmation_steps": DEFAULT_CFG["confirmation_steps"]}
    sub = df.copy()
    for p, v in hold.items():
        if p == vary:
            continue
        if p in sub.columns and sub[p].nunique() > 1:
            # fall back to the most common value if the exact default is absent
            val = v if (sub[p] == v).any() else sub[p].mode().iloc[0]
            sub = sub[sub[p] == val]
    return sub


def _seed_stats(sub, xcol, ycol):
    g = sub.groupby(xcol)[ycol]
    return g.mean(), g.std().fillna(0.0)


def make_sweep_figures(df, out_dir):
    saved = []

    # Fig 1 — detection rate vs Q
    sub = _slice_holding_defaults(df, "Q")
    if sub["Q"].nunique() > 1:
        mean, std = _seed_stats(sub, "Q", "n_detected")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(mean.index, mean.values, yerr=std.values, marker="o", capsize=4, color="#185FA5")
        ax.axhline(sub["n_drifts"].iloc[0], ls="--", color="#A32D2D", label="all drifts")
        ax.set_xlabel("Series growth exponent  Q"); ax.set_ylabel("Drifts detected (seed mean ± std)")
        ax.set_title("Detection rate vs Q  (τ, w, k, γ at primary)"); ax.grid(alpha=0.3); ax.legend()
        _save(fig, out_dir, "sweep_fig1_detection_vs_Q.png", saved)

    # Fig 2 — mean delay vs Q
    sub = _slice_holding_defaults(df, "Q")
    if sub["Q"].nunique() > 1 and sub["delay_mean"].notna().any():
        mean, std = _seed_stats(sub, "Q", "delay_mean")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(mean.index, mean.values, yerr=std.values, marker="s", capsize=4, color="#0F6E56")
        ax.set_xlabel("Series growth exponent  Q"); ax.set_ylabel("Mean detection delay (samples)")
        ax.set_title("Detection delay vs Q  (seed mean ± std)"); ax.grid(alpha=0.3)
        _save(fig, out_dir, "sweep_fig2_delay_vs_Q.png", saved)

    # Fig 3 — alarms vs tau
    sub = _slice_holding_defaults(df, "tau")
    if sub["tau"].nunique() > 1:
        tot_m, tot_s = _seed_stats(sub, "tau", "total_alarms")
        un_m, un_s = _seed_stats(sub, "tau", "unmatched_alarms")
        det_m, _ = _seed_stats(sub, "tau", "n_detected")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(tot_m.index, tot_m.values, yerr=tot_s.values, marker="o", capsize=4, label="total alarms", color="#185FA5")
        ax.errorbar(un_m.index, un_m.values, yerr=un_s.values, marker="^", capsize=4, label="unmatched (FP)", color="#D85A30")
        ax.set_xlabel("ISE threshold  τ"); ax.set_ylabel("Alarms (seed mean ± std)")
        ax2 = ax.twinx(); ax2.plot(det_m.index, det_m.values, "--", color="#3B6D11", label="drifts detected")
        ax2.set_ylabel("Drifts detected", color="#3B6D11")
        ax.set_title("Alarm counts vs ISE threshold τ"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
        _save(fig, out_dir, "sweep_fig3_alarms_vs_tau.png", saved)

    # Fig 4 — detection rate vs warmup
    sub = _slice_holding_defaults(df, "warmup")
    if sub["warmup"].nunique() > 1:
        mean, std = _seed_stats(sub, "warmup", "n_detected")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(mean.index, mean.values, yerr=std.values, marker="o", capsize=4, color="#534AB7")
        ax.set_xlabel("Warm-up length  w"); ax.set_ylabel("Drifts detected (seed mean ± std)")
        ax.set_title("Detection rate vs warm-up length"); ax.grid(alpha=0.3)
        _save(fig, out_dir, "sweep_fig4_detection_vs_warmup.png", saved)

    # Fig 4b — detection AND false alarms vs confirmation window c
    sub = _slice_holding_defaults(df, "confirmation_steps")
    if "confirmation_steps" in sub.columns and sub["confirmation_steps"].nunique() > 1:
        det_m, det_s = _seed_stats(sub, "confirmation_steps", "n_detected")
        fa_m, fa_s = _seed_stats(sub, "confirmation_steps", "unmatched_alarms")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(det_m.index, det_m.values, yerr=det_s.values, marker="o", capsize=4,
                    color="#185FA5", label="drifts detected")
        ax.set_xlabel("Confirmation window  c"); ax.set_ylabel("Drifts detected", color="#185FA5")
        ax2 = ax.twinx()
        ax2.errorbar(fa_m.index, fa_m.values, yerr=fa_s.values, marker="^", capsize=4,
                     color="#D85A30", label="false alarms")
        ax2.set_ylabel("False alarms (unmatched)", color="#D85A30")
        ax.set_title("Effect of confirmation window c on detection and false alarms")
        ax.grid(alpha=0.3)
        _save(fig, out_dir, "sweep_fig4b_detection_falsealarms_vs_c.png", saved)

    # Fig 5 — heatmap detection rate over Q x tau
    sub = df.copy()
    for p in ("warmup", "k", "gamma", "confirmation_steps"):
        if p in sub.columns and sub[p].nunique() > 1:
            v = DEFAULT_CFG[p] if (sub[p] == DEFAULT_CFG[p]).any() else sub[p].mode().iloc[0]
            sub = sub[sub[p] == v]
    if sub["Q"].nunique() > 1 and sub["tau"].nunique() > 1:
        piv = sub.groupby(["Q", "tau"])["n_detected"].mean().unstack("tau")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("ISE threshold  τ"); ax.set_ylabel("Series growth exponent  Q")
        ax.set_title("Mean drifts detected over (Q, τ)")
        for ii in range(piv.shape[0]):
            for jj in range(piv.shape[1]):
                val = piv.values[ii, jj]
                if not np.isnan(val):
                    ax.text(jj, ii, f"{val:.0f}", ha="center", va="center",
                            color="white" if val < piv.values[~np.isnan(piv.values)].max() * 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="drifts detected")
        _save(fig, out_dir, "sweep_fig5_heatmap_Q_tau.png", saved)

    # Fig 6 — seed variability of mean delay at the primary configuration
    prim = df.copy()
    for p, v in (("Q", DEFAULT_CFG["Q"]), ("tau", DEFAULT_CFG["tau"]), ("warmup", DEFAULT_CFG["warmup"]),
                 ("k", DEFAULT_CFG["k"]), ("gamma", DEFAULT_CFG["gamma"]),
                 ("confirmation_steps", DEFAULT_CFG["confirmation_steps"])):
        if p in prim.columns and (prim[p] == v).any():
            prim = prim[prim[p] == v]
    if len(prim) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot([prim["delay_mean"].dropna(), prim["delay_median"].dropna()])
        ax.set_xticks([1, 2]); ax.set_xticklabels(["mean delay", "median delay"])
        for i, col in enumerate(["delay_mean", "delay_median"], start=1):
            ax.scatter(np.full(prim[col].notna().sum(), i), prim[col].dropna(), color="#185FA5", alpha=0.7, zorder=3)
        ax.set_ylabel("Detection delay (samples)")
        ax.set_title(f"Seed variability at primary config (n={len(prim)} seeds)"); ax.grid(alpha=0.3)
        _save(fig, out_dir, "sweep_fig6_seed_variability.png", saved)

    # Fig 7 — detection vs false alarms trade-off (the sweet-spot view)
    g = (df.groupby(["Q", "tau", "warmup", "k", "gamma", "confirmation_steps"])
           .agg(detected=("n_detected", "mean"),
                false_alarms=("unmatched_alarms", "mean"),
                drifts=("n_drifts", "mean"))
           .reset_index())
    if len(g) > 1:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sc = ax.scatter(g["false_alarms"], g["detected"], c=g["tau"], cmap="plasma",
                        s=45, edgecolor="k", linewidth=0.3, alpha=0.85)
        fig.colorbar(sc, ax=ax, label="ISE threshold  τ")
        # best = most detected, then fewest false alarms
        gb = g.sort_values(["detected", "false_alarms"], ascending=[False, True]).iloc[0]
        ax.scatter([gb["false_alarms"]], [gb["detected"]], s=260, facecolors="none",
                   edgecolors="#1a9850", linewidths=2.2, zorder=5)
        ax.annotate(f"best: Q={gb['Q']}, k={gb['k']}, γ={gb['gamma']},\n"
                    f"w={int(gb['warmup'])}, τ={gb['tau']}, c={int(gb['confirmation_steps'])}\n"
                    f"{gb['detected']:.1f}/{gb['drifts']:.0f} detected, "
                    f"{gb['false_alarms']:.1f} false alarms",
                    (gb["false_alarms"], gb["detected"]),
                    textcoords="offset points", xytext=(12, -10), fontsize=8,
                    bbox=dict(boxstyle="round", fc="#eaf7ea", ec="#1a9850", alpha=0.9))
        ax.set_xlabel("False alarms — unmatched alarms (seed mean)")
        ax.set_ylabel("Drifts detected (seed mean)")
        ax.set_title("Detection vs false alarms across all configurations\n"
                     "(top-left = detects everything with few false alarms)")
        ax.grid(alpha=0.3)
        _save(fig, out_dir, "sweep_fig7_detection_vs_falsealarms.png", saved)

    print(f"  Generated {len(saved)} sweep figures: {', '.join(os.path.basename(s) for s in saved)}")
    return saved


def _save(fig, out_dir, name, saved):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def _parse_args():
    p = argparse.ArgumentParser(description="Full-grid + multi-seed sweep for the IPNN drift detector.")
    p.add_argument("--quick", action="store_true", help="Tiny grid x 2 seeds for a fast check.")
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Override the seed list.")
    p.add_argument("--n", type=int, default=None, help="Override stream length N (must exceed 15000).")
    p.add_argument("--out", type=str, default=None, help="Output folder.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel worker processes. Use -1 for all CPU cores. Default 1 (sequential).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to an existing sweep_results.csv; already-finished runs are skipped.")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    grid = QUICK_GRID if a.quick else GRID
    seeds = a.seeds if a.seeds else (QUICK_SEEDS if a.quick else SEEDS)
    out = a.out or f"sweeps/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = (os.cpu_count() or 1) if a.jobs in (-1, 0) else a.jobs
    run_sweep(grid, seeds, out, n_override=a.n, confirm=not a.yes, jobs=jobs, resume=a.resume)
