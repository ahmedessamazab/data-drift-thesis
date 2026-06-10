"""
Full Experiment — Multiple Drift Points + CSV Exports  (refactored)
===================================================================
This module now exposes a reusable ``run_experiment(cfg)`` function so the
same detection pipeline can be driven either:

  * as a SINGLE run from the command line (reproduces the thesis primary
    result and writes the 3 plots + 6 CSVs + experiment_config.json), or
  * programmatically by ``run_sweep.py`` over a full grid of hyper-parameters
    and multiple random seeds.

Running this file directly with no arguments reproduces the thesis primary
configuration (Q=0.5, k=3.0, gamma=1.0, warm-up=300, tau=0.08, c=7, N=20000,
seed=42).  Every parameter can be overridden from the command line, e.g.:

    python run_full_experiment.py --Q 0.4 --tau 0.085 --seed 7

PLOTS (single-run mode)
  01_stream_overview.png   02_detection_report.png   03_pdf_evolution.png
CSV FILES (single-run mode)
  csv_01_stream_raw.csv ... csv_06_pdf_snapshots.csv   (+ experiment_config.json)

FILES YOU NEED (same folder): series.py, ipnn.py, drift_detector.py,
synthetic_stream.py, run_full_experiment.py
LIBRARIES: pip install numpy matplotlib pandas
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so sweeps run head-less without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import series
from ipnn import IPNN
from drift_detector import DriftDetector
from synthetic_stream import (
    SyntheticStreamGenerator,
    DriftSpec,
    plot_stream,
    plot_detection_report,
)

# ══════════════════════════════════════════════════════════════════════════════
# Default configuration  (thesis primary experiment)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CFG = {
    "N": 20000,
    "seed": 42,
    "Q": 0.5,            # series-growth exponent  q(n) = k * (n+1)^Q
    "k": 3.0,            # series-growth scale
    "gamma": 1.0,        # learning-rate exponent  gamma_n = (n+1)^-gamma
    "warmup": 300,       # warm-up length w
    "tau": 0.08,         # ISE detection threshold
    "confirmation_steps": 7,   # consecutive exceedances required to confirm (c)
    "net_min": -4.0,
    "net_max": 10.0,
    "net_size": 400,
    "kernel": "Hermite",
}


def build_drift_specs():
    """Return the 11-event drift schedule used throughout the thesis."""
    return [
        DriftSpec(position=800,   drift_type="mean",         mean_before=0.0, std_before=1.0, mean_after=2.5, std_after=1.0, label="Abrupt mean drift (0 to 2.5)"),
        DriftSpec(position=1500,  drift_type="variance",     mean_before=2.5, std_before=1.0, mean_after=2.5, std_after=2.5, label="Variance drift (sigma 1 to 2.5)"),
        DriftSpec(position=2200,  drift_type="gradual",      mean_before=2.5, std_before=2.5, mean_after=5.0, std_after=1.0, transition_width=200, label="Gradual mean drift (2.5 to 5.0)"),
        DriftSpec(position=3000,  drift_type="variance",     mean_before=5.0, std_before=1.0, mean_after=5.0, std_after=0.3, label="Variance shrink (1.0 -> 0.3)"),
        DriftSpec(position=4000,  drift_type="mean",         mean_before=5.0, std_before=0.3, mean_after=0.0, std_after=1.0, label="Mean reversal (5 -> 0)"),
        DriftSpec(position=5000,  drift_type="cyclic",       mean_before=0.0, std_before=1.0, mean_after=0.0, std_after=1.0, label="Cyclic drift (sin wave)"),
        DriftSpec(position=6000,  drift_type="distribution", mean_before=0.0, std_before=1.0, mean_after=0.0, std_after=1.0, label="Gaussian -> Uniform"),
        DriftSpec(position=7000,  drift_type="gradual",      mean_before=0.0, std_before=1.0, mean_after=3.0, std_after=1.0, transition_width=800, label="Very slow gradual drift"),
        DriftSpec(position=9000,  drift_type="mean",         mean_before=3.0, std_before=1.0, mean_after=2.5, std_after=1.0, label="Abrupt mean drift 2 (3.0 to 2.5)"),
        DriftSpec(position=12000, drift_type="variance",     mean_before=2.5, std_before=1.0, mean_after=2.5, std_after=3.0, label="Late variance expansion"),
        DriftSpec(position=15000, drift_type="variance",     mean_before=2.5, std_before=3.0, mean_after=2.5, std_after=0.5, label="Noise reduction"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Fast density evaluation
# ══════════════════════════════════════════════════════════════════════════════
def build_basis_matrix(net, kernel, q_max):
    """
    Pre-compute the orthonormal basis matrix B with B[m, j] = phi_j(net[m]).

    The grid ``net`` is fixed for the whole run, so the basis values never
    change — only the active order q and the coefficients do.  Evaluating the
    density then reduces to a single matrix-vector product
        pdf = B[:, :q] @ a_j[:q]
    which is dramatically faster than calling the basis function at every grid
    point on every step (the original per-step Python loop).
    """
    fn = series.choices[kernel]
    M = len(net)
    B = np.zeros((M, q_max))
    for m in range(M):
        row = fn(float(net[m]), q_max)  # returns length q_max + 1
        B[m, :] = row[:q_max]
    return B


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════
def match_alarms_to_drifts(drift_specs, alarm_positions, N):
    """
    Match each true drift to the first confirmed alarm that falls WITHIN its
    detection window [drift_position, next_drift_position) -- the slot definition
    stated in the thesis methodology (Section 5.4). An alarm that arrives only
    after the next drift has already started does NOT count as detecting the
    earlier drift; it is left as an unmatched (false) alarm. The window for the
    final drift runs to the end of the stream, [last_position, N).
    """
    positions = [s.position for s in drift_specs]
    alarms_sorted = sorted(alarm_positions)
    results = []
    used = set()
    for i, spec in enumerate(drift_specs):
        lo = spec.position
        hi = positions[i + 1] if i + 1 < len(positions) else N
        candidates = [a for a in alarms_sorted if lo <= a < hi and a not in used]
        if candidates:
            alarm_at = candidates[0]
            used.add(alarm_at)
            results.append({"drift_id": i + 1, "true_position": spec.position, "type": spec.drift_type,
                            "detected": True, "alarm_position": alarm_at,
                            "delay_samples": alarm_at - spec.position, "label": spec.label})
        else:
            results.append({"drift_id": i + 1, "true_position": spec.position, "type": spec.drift_type,
                            "detected": False, "alarm_position": None, "delay_samples": None, "label": spec.label})
    return results, used


def summarise(results, used, alarm_positions, ise_history, cfg):
    """Compute the scalar metrics used in the comparative analysis."""
    n_drifts = len(results)
    detected = [r for r in results if r["detected"]]
    delays = [r["delay_samples"] for r in detected]

    by_type = defaultdict(list)
    for r in detected:
        by_type[r["type"]].append(r["delay_samples"])

    ise = np.asarray(ise_history, dtype=float)
    active = ise[cfg["warmup"] + 1:] if len(ise) > cfg["warmup"] + 1 else ise

    m = {
        "N": cfg["N"], "seed": cfg["seed"], "Q": cfg["Q"], "k": cfg["k"],
        "gamma": cfg["gamma"], "warmup": cfg["warmup"], "tau": cfg["tau"],
        "confirmation_steps": cfg["confirmation_steps"],
        "n_drifts": n_drifts,
        "n_detected": len(detected),
        "tpr": len(detected) / n_drifts if n_drifts else 0.0,
        "total_alarms": len(alarm_positions),
        "matched_alarms": len(used),
        "unmatched_alarms": len(alarm_positions) - len(used),
        "unmatched_ratio": (len(alarm_positions) - len(used)) / len(alarm_positions) if alarm_positions else 0.0,
        "delay_min": min(delays) if delays else None,
        "delay_max": max(delays) if delays else None,
        "delay_mean": float(np.mean(delays)) if delays else None,
        "delay_median": float(np.median(delays)) if delays else None,
        "max_ise": float(np.max(ise)) if len(ise) else 0.0,
        "mean_background_ise": float(np.mean(active)) if len(active) else 0.0,
        "threshold_exceedances": int(np.sum(ise > cfg["tau"])),
    }
    # Per-drift-type mean delay. NOTE the 'delaytype_' prefix: using 'delay_mean'
    # here would collide with the OVERALL mean-delay key above.
    for dtype in ("mean", "variance", "gradual", "cyclic", "distribution"):
        vals = by_type.get(dtype, [])
        m[f"delaytype_{dtype}"] = float(np.mean(vals)) if vals else None
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(cfg=None, out_dir=None, make_plots=True, save_csvs=True, verbose=True):
    """
    Run one IPNN drift-detection experiment.

    Parameters
    ----------
    cfg : dict
        Configuration; missing keys fall back to ``DEFAULT_CFG``.
    out_dir : str or None
        Folder for plots/CSVs.  If None and outputs are requested, a timestamped
        folder is created under ``experiments/``.
    make_plots, save_csvs : bool
        Set both to False for fast sweep runs (metrics only).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Scalar metrics for the run (see ``summarise``), plus the key arrays
        under 'results', 'ise_history' and 'alarm_positions'.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    drift_specs = build_drift_specs()
    N = cfg["N"]
    net = np.linspace(cfg["net_min"], cfg["net_max"], cfg["net_size"])

    need_dir = (make_plots or save_csvs)
    if need_dir and out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        names = "_".join(d.drift_type for d in drift_specs)
        out_dir = f"experiments/{ts}_N{N}_drifts{len(drift_specs)}_{names}"
    if need_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- stream ---------------------------------------------------------------
    if verbose:
        print(f"\n[1/4] Generating synthetic stream (N={N}, seed={cfg['seed']}) ...")
    gen = SyntheticStreamGenerator(seed=cfg["seed"])
    stream = gen.generate(N, drift_specs)

    # --- model + detector -----------------------------------------------------
    model = IPNN(method="series", kernel=cfg["kernel"], Q=cfg["Q"], k=cfg["k"], gamma=cfg["gamma"])
    detector = DriftDetector(net_of_x=net, threshold=cfg["tau"], warmup=cfg["warmup"],
                             confirmation_steps=cfg["confirmation_steps"])

    q_max = int(cfg["k"] * math.pow(N, cfg["Q"])) + 2
    B = build_basis_matrix(net, cfg["kernel"], q_max)

    alarm_positions = []
    coeff_rows = []
    pdf_snapshots = {}
    snapshot_times = {cfg["warmup"], N - 1}
    if make_plots or save_csvs:
        for spec in drift_specs:
            for off in (-1, 100, 200):
                t = spec.position + off
                if 0 <= t < N:
                    snapshot_times.add(t)

    if verbose:
        print("[2/4] Running IPNN detector ...")
    for n, x_new in enumerate(stream.data):
        model.update_aj(x_new, n, 1.0)
        q = int(model.q(n))
        if q > q_max:                      # safety (should not happen)
            q = q_max
        current_pdf = B[:, :q] @ model.a_j[:q, 0]

        if make_plots or save_csvs:
            a0 = float(model.a_j[0, 0]) if len(model.a_j) > 0 else 0.0
            a1 = float(model.a_j[1, 0]) if len(model.a_j) > 1 else 0.0
            coeff_rows.append({"index": n, "a0": a0, "a1": a1})
            if n in snapshot_times:
                pdf_snapshots[n] = current_pdf.copy()

        if detector.update(current_pdf, n):
            alarm_positions.append(n)
            detector.reference_pdf = current_pdf.copy()
            detector.alarm_index = None

        if verbose and N >= 10 and (n + 1) % (N // 10) == 0:
            print(f"      {100 * (n + 1) // N:3d}%  alarms so far: {len(alarm_positions)}")

    # --- metrics --------------------------------------------------------------
    results, used = match_alarms_to_drifts(drift_specs, alarm_positions, N)
    metrics = summarise(results, used, alarm_positions, detector.ise_history, cfg)
    metrics["results"] = results
    metrics["ise_history"] = detector.ise_history
    metrics["alarm_positions"] = alarm_positions

    if verbose:
        print(f"      detected {metrics['n_detected']}/{metrics['n_drifts']}  | "
              f"alarms {metrics['total_alarms']} ({metrics['unmatched_alarms']} unmatched) | "
              f"mean delay {metrics['delay_mean']}")

    # --- optional outputs (single-run mode) -----------------------------------
    if save_csvs or make_plots:
        _write_outputs(out_dir, cfg, net, stream, drift_specs, results,
                       detector, alarm_positions, coeff_rows, pdf_snapshots,
                       save_csvs, make_plots, verbose)
        metrics["out_dir"] = out_dir
    return metrics


def _write_outputs(out_dir, cfg, net, stream, drift_specs, results, detector,
                   alarm_positions, coeff_rows, pdf_snapshots, save_csvs, make_plots, verbose):
    """Reproduce the original 6 CSVs + 3 plots + experiment_config.json."""
    N = cfg["N"]

    def save_csv(df, name):
        df.to_csv(os.path.join(out_dir, name), index=False)
        if verbose:
            print(f"  saved -> {os.path.join(out_dir, name)}")

    def save_fig(fig, name):
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  saved -> {os.path.join(out_dir, name)}")

    if save_csvs:
        seg_mean = np.zeros(N); seg_std = np.zeros(N); seg_id = np.zeros(N, dtype=int)
        boundaries = [0] + stream.true_drift_positions + [N]
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            seg_id[a:b] = i
            seg_mean[a:b] = drift_specs[0].mean_before if i == 0 else drift_specs[i - 1].mean_after
            seg_std[a:b] = drift_specs[0].std_before if i == 0 else drift_specs[i - 1].std_after

        save_csv(pd.DataFrame({"index": np.arange(N), "value": stream.data, "segment_id": seg_id,
                               "segment_label": stream.segment_labels, "nominal_mean": seg_mean,
                               "nominal_std": seg_std}), "csv_01_stream_raw.csv")
        save_csv(pd.DataFrame([{"drift_id": i + 1, "position": s.position, "type": s.drift_type,
                                "mean_before": s.mean_before, "std_before": s.std_before,
                                "mean_after": s.mean_after, "std_after": s.std_after, "label": s.label}
                               for i, s in enumerate(drift_specs)]), "csv_02_drift_ground_truth.csv")

        ise_arr = np.array(detector.ise_history)
        alarm_set = set(alarm_positions)
        save_csv(pd.DataFrame({"index": np.arange(len(ise_arr)), "ise_score": ise_arr,
                               "alarm_fired": [1 if i in alarm_set else 0 for i in range(len(ise_arr))],
                               "above_threshold": (ise_arr > cfg["tau"]).astype(int)}), "csv_03_ise_score.csv")
        save_csv(pd.DataFrame(coeff_rows), "csv_04_coefficients.csv")
        save_csv(pd.DataFrame(results), "csv_05_detection_results.csv")

        df_pdf = pd.DataFrame({"x": net})
        for t, pdf in sorted(pdf_snapshots.items()):
            col = f"n{t}"
            if t == cfg["warmup"]:
                col += "_reference"
            for j, spec in enumerate(drift_specs):
                if t == spec.position - 1:   col += f"_pre_drift{j+1}"
                if t == spec.position + 100: col += f"_post_drift{j+1}"
                if t == spec.position + 200: col += f"_post_drift{j+1}_200"
            if t == N - 1:
                col += "_final"
            df_pdf[col] = pdf
        save_csv(df_pdf, "csv_06_pdf_snapshots.csv")

    if make_plots:
        fig = plot_stream(stream, window=60, title="Synthetic stream — known drift locations (dashed lines)")
        save_fig(fig, "01_stream_overview.png")
        fig_report, _ = plot_detection_report(stream, alarm_positions=alarm_positions,
                                               ise_history=detector.ise_history, threshold=cfg["tau"],
                                               title="Detection report — caught (green dotted) vs missed")
        save_fig(fig_report, "02_detection_report.png")

        fig3, ax = plt.subplots(figsize=(12, 6))
        fig3.suptitle("PDF evolution at key time-steps (Hermite series)", fontsize=13)
        palette = ["#534AB7", "#0F6E56", "#D85A30", "#185FA5", "#993556", "#BA7517", "#3B6D11", "#A32D2D"]
        for (t, pdf), col in zip(sorted(pdf_snapshots.items()), palette):
            lbl = f"n={t}"
            if t == cfg["warmup"]:
                lbl += " (reference)"
            if t == N - 1:
                lbl += " (final)"
            ax.plot(net, pdf, color=col, linewidth=1.4, label=lbl)
        ax.set_xlabel("x"); ax.set_ylabel("Density"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        save_fig(fig3, "03_pdf_evolution.png")

    config = {"N": N, "SEED": cfg["seed"], "Q_PARAM": cfg["Q"], "K_PARAM": cfg["k"],
              "GAMMA": cfg["gamma"], "WARMUP": cfg["warmup"], "ISE_THRESHOLD": cfg["tau"],
              "CONFIRMATION_STEPS": cfg["confirmation_steps"], "NET_SIZE": cfg["net_size"],
              "DRIFTS": [{"position": d.position, "type": d.drift_type, "mean_before": d.mean_before,
                          "std_before": d.std_before, "mean_after": d.mean_after, "std_after": d.std_after,
                          "label": d.label} for d in drift_specs]}
    with open(os.path.join(out_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
# Command-line single-run entry point
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(description="Run a single IPNN drift-detection experiment.")
    p.add_argument("--Q", type=float, default=DEFAULT_CFG["Q"])
    p.add_argument("--k", type=float, default=DEFAULT_CFG["k"])
    p.add_argument("--gamma", type=float, default=DEFAULT_CFG["gamma"])
    p.add_argument("--warmup", type=int, default=DEFAULT_CFG["warmup"])
    p.add_argument("--tau", type=float, default=DEFAULT_CFG["tau"])
    p.add_argument("--confirm", type=int, default=DEFAULT_CFG["confirmation_steps"])
    p.add_argument("--N", type=int, default=DEFAULT_CFG["N"])
    p.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    p.add_argument("--out", type=str, default=None, help="Output folder (default: timestamped under experiments/)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = {"Q": a.Q, "k": a.k, "gamma": a.gamma, "warmup": a.warmup, "tau": a.tau,
           "confirmation_steps": a.confirm, "N": a.N, "seed": a.seed}
    m = run_experiment(cfg, out_dir=a.out, make_plots=not a.no_plots, save_csvs=True, verbose=True)
    print("\n" + "=" * 70)
    print(f"  detected {m['n_detected']}/{m['n_drifts']}  (TPR {100*m['tpr']:.0f}%)")
    print(f"  total alarms {m['total_alarms']}  | unmatched {m['unmatched_alarms']}")
    print(f"  delay  min {m['delay_min']}  median {m['delay_median']}  mean {m['delay_mean']}  max {m['delay_max']}")
    print(f"  outputs -> {m.get('out_dir')}")
    print("=" * 70)
