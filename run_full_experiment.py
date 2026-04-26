"""
Full Experiment — Multiple Drift Points + CSV Exports
======================================================
Produces:

PLOTS
-----
  01_stream_overview.png       Raw stream + rolling stats + segment map
  02_detection_report.png      ISE score + which drifts were caught/missed
  03_pdf_evolution.png         How the estimated PDF changes over time

CSV FILES
---------
  csv_01_stream_raw.csv          Every sample: index, value, segment info
  csv_02_drift_ground_truth.csv  The injected drift events (ground truth)
  csv_03_ise_score.csv           ISE score at every time step
  csv_04_coefficients.csv        First two Hermite coefficients over time
  csv_05_detection_results.csv   Final report — one row per drift event
  csv_06_pdf_snapshots.csv       Full PDF curves at key moments

FILES YOU NEED (put all in the same folder)
--------------------------------------------
  series.py
  ipnn.py
  drift_detector.py
  synthetic_stream.py
  run_full_experiment.py   <- this file

LIBRARIES TO INSTALL
--------------------
  pip install numpy matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from datetime import datetime


from series import choices as series_choices
from ipnn import IPNN
from drift_detector import DriftDetector
from synthetic_stream import (
    SyntheticStreamGenerator,
    DriftSpec,
    plot_stream,
    plot_detection_report,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
N = 20000

DRIFT_SPECS = [
    DriftSpec(
        position=800,
        drift_type="mean",
        mean_before=0.0,
        std_before=1.0,
        mean_after=2.5,
        std_after=1.0,
        label="Abrupt mean drift (0 to 2.5)",
    ),
    DriftSpec(
        position=1500,
        drift_type="variance",
        mean_before=2.5,
        std_before=1.0,
        mean_after=2.5,
        std_after=2.5,
        label="Variance drift (sigma 1 to 2.5)",
    ),
    DriftSpec(
        position=2200,
        drift_type="gradual",
        mean_before=2.5,
        std_before=2.5,
        mean_after=5.0,
        std_after=1.0,
        transition_width=200,
        label="Gradual mean drift (2.5 to 5.0)",
    ),
    DriftSpec(
        position=3000,
        drift_type="variance",
        mean_before=5.0,
        std_before=1.0,
        mean_after=5.0,
        std_after=0.3,
        label="Variance shrink (1.0 → 0.3)",
    ),
    DriftSpec(
        position=4000,
        drift_type="mean",
        mean_before=5.0,
        std_before=0.3,
        mean_after=0.0,
        std_after=1.0,
        label="Mean reversal (5 → 0)",
    ),
    DriftSpec(
        position=5000,
        drift_type="cyclic",
        mean_before=0.0,
        std_before=1.0,
        mean_after=0.0,
        std_after=1.0,
        label="Cyclic drift (sin wave)",
    ),
    DriftSpec(
        position=6000,
        drift_type="distribution",
        mean_before=0.0,
        std_before=1.0,
        mean_after=0.0,
        std_after=1.0,
        label="Gaussian → Uniform",
    ),
    DriftSpec(
        position=7000,
        drift_type="gradual",
        mean_before=0.0,
        std_before=1.0,
        mean_after=3.0,
        std_after=1.0,
        transition_width=800,
        label="Very slow gradual drift",
    ),
    DriftSpec(
        position=9000,
        drift_type="mean",
        mean_before=3.0,  # ✅ FIXED
        std_before=1.0,
        mean_after=2.5,
        std_after=1.0,
        label="Abrupt mean drift 2 (3.0 to 2.5)",
    ),
    DriftSpec(
        position=12000,
        drift_type="variance",
        mean_before=2.5,
        std_before=1.0,
        mean_after=2.5,
        std_after=3.0,
        label="Late variance expansion",
    ),
    DriftSpec(
        position=15000,
        drift_type="variance",
        mean_before=2.5,
        std_before=3.0,
        mean_after=2.5,
        std_after=0.5,
        label="Noise reduction",
    ),
]

Q_PARAM = 0.5
K_PARAM = 3.0
GAMMA = 1.0
WARMUP = 300
ISE_THRESHOLD = 0.1
CONFIRMATION_STEPS = 7  # Number of consecutive steps above threshold to confirm drift
NET = np.linspace(-4, 10, 400)


# Helper to build experiment name based on config
def build_experiment_name():
    drift_names = "_".join([d.drift_type for d in DRIFT_SPECS])
    return f"N{N}_drifts{len(DRIFT_SPECS)}_{drift_names}"


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT = f"experiments/{timestamp}_{build_experiment_name()}"
os.makedirs(OUT, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def save_csv(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"  saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate synthetic stream + CSV + plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Generating synthetic stream ...")

gen = SyntheticStreamGenerator(seed=SEED)
stream = gen.generate(N, DRIFT_SPECS)
print(f"      {N} samples, drift points: {stream.true_drift_positions}")

# Per-sample segment statistics
seg_mean = np.zeros(N)
seg_std = np.zeros(N)
seg_id = np.zeros(N, dtype=int)
boundaries = [0] + stream.true_drift_positions + [N]
# Segment statistics represent nominal post-drift parameters and may not reflect dynamic drift types such as cyclic or gradual transitions.
for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    seg_id[a:b] = i
    seg_mean[a:b] = (
        DRIFT_SPECS[0].mean_before if i == 0 else DRIFT_SPECS[i - 1].mean_after
    )
    seg_std[a:b] = DRIFT_SPECS[0].std_before if i == 0 else DRIFT_SPECS[i - 1].std_after

# CSV 01 — raw stream
save_csv(
    pd.DataFrame(
        {
            "index": np.arange(N),
            "value": stream.data,
            "segment_id": seg_id,
            "segment_label": stream.segment_labels,
            "nominal_mean": seg_mean,
            "nominal_std": seg_std,
        }
    ),
    "csv_01_stream_raw.csv",
)

# CSV 02 — ground truth drifts
save_csv(
    pd.DataFrame(
        [
            {
                "drift_id": i + 1,
                "position": s.position,
                "type": s.drift_type,
                "mean_before": s.mean_before,
                "std_before": s.std_before,
                "mean_after": s.mean_after,
                "std_after": s.std_after,
                "label": s.label,
            }
            for i, s in enumerate(DRIFT_SPECS)
        ]
    ),
    "csv_02_drift_ground_truth.csv",
)

fig = plot_stream(
    stream, window=60, title="Synthetic stream — known drift locations (dashed lines)"
)
save_fig(fig, "01_stream_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run IPNN + DriftDetector
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Running IPNN detector ...")

model = IPNN(method="series", kernel="Hermite", Q=Q_PARAM, k=K_PARAM, gamma=GAMMA)
detector = DriftDetector(
    net_of_x=NET,
    threshold=ISE_THRESHOLD,
    warmup=WARMUP,
    confirmation_steps=CONFIRMATION_STEPS,
)

alarm_positions = []
coeff_rows = []
pdf_snapshots = {}

SNAPSHOT_TIMES = {WARMUP, N - 1}

for spec in DRIFT_SPECS:
    if spec.position - 1 >= 0:
        SNAPSHOT_TIMES.add(spec.position - 1)

    if spec.position + 100 < N:
        SNAPSHOT_TIMES.add(spec.position + 100)

    if spec.position + 200 < N:
        SNAPSHOT_TIMES.add(spec.position + 200)

for n, x_new in enumerate(stream.data):
    model.update_aj(x_new, n, 1.0)
    q = int(model.q(n))

    current_pdf = np.array(
        [float(np.inner(model.ker(xi, q)[:q], model.a_j[:q, 0])) for xi in NET]
    )

    a0 = float(model.a_j[0, 0]) if len(model.a_j) > 0 else 0.0
    a1 = float(model.a_j[1, 0]) if len(model.a_j) > 1 else 0.0
    coeff_rows.append({"index": n, "a0": a0, "a1": a1})

    if n in SNAPSHOT_TIMES:
        pdf_snapshots[n] = current_pdf.copy()

    alarm_fired = detector.update(current_pdf, n)
    if alarm_fired:
        alarm_positions.append(n)
        detector.reference_pdf = current_pdf.copy()
        detector.alarm_index = None

    if (n + 1) % (N // 10) == 0:
        print(f"      {100*(n+1)//N:3d}%  alarms so far: {alarm_positions}")

print(f"      Finished. Total alarms raised: {len(alarm_positions)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Export process CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Exporting process CSVs ...")

ise_arr = np.array(detector.ise_history)
alarm_set = set(alarm_positions)

# CSV 03 — ISE score at every step
save_csv(
    pd.DataFrame(
        {
            "index": np.arange(len(ise_arr)),
            "ise_score": ise_arr,
            "alarm_fired": [1 if i in alarm_set else 0 for i in range(len(ise_arr))],
            "above_threshold": (ise_arr > ISE_THRESHOLD).astype(int),
        }
    ),
    "csv_03_ise_score.csv",
)

# CSV 04 — coefficients over time
save_csv(pd.DataFrame(coeff_rows), "csv_04_coefficients.csv")

# CSV 05 — detection results (match each true drift to nearest subsequent alarm)
results = []
used = set()
for i, spec in enumerate(DRIFT_SPECS):
    candidates = [
        a for a in sorted(alarm_positions) if a >= spec.position and a not in used
    ]
    if candidates:
        alarm_at = candidates[0]
        used.add(alarm_at)
        results.append(
            {
                "drift_id": i + 1,
                "true_position": spec.position,
                "type": spec.drift_type,
                "detected": True,
                "alarm_position": alarm_at,
                "delay_samples": alarm_at - spec.position,
                "label": spec.label,
            }
        )
    else:
        results.append(
            {
                "drift_id": i + 1,
                "true_position": spec.position,
                "type": spec.drift_type,
                "detected": False,
                "alarm_position": None,
                "delay_samples": None,
                "label": spec.label,
            }
        )

save_csv(pd.DataFrame(results), "csv_05_detection_results.csv")

# CSV 06 — PDF snapshots (one column per time-step)
df_pdf = pd.DataFrame({"x": NET})
for t, pdf in sorted(pdf_snapshots.items()):
    col = f"n{t}"
    if t == WARMUP:
        col += "_reference"
    for j, spec in enumerate(DRIFT_SPECS):
        if t == spec.position - 1:
            col += f"_pre_drift{j+1}"
        if t == spec.position + 100:
            col += f"_post_drift{j+1}"
        if t == spec.position + 200:
            col += f"_post_drift{j+1}_200"
    if t == N - 1:
        col += "_final"
    df_pdf[col] = pdf
save_csv(df_pdf, "csv_06_pdf_snapshots.csv")

# Print detection table
print()
print("  +---------+----------+-----------+-----------+----------+---------+")
print("  | Drift   | True pos | Type      | Detected? | Alarm at | Delay   |")
print("  +---------+----------+-----------+-----------+----------+---------+")
for r in results:
    det = "YES" if r["detected"] else "NO "
    alm = str(r["alarm_position"]) if r["alarm_position"] is not None else "---"
    dly = str(r["delay_samples"]) if r["delay_samples"] is not None else "---"
    print(
        f"  | {r['drift_id']:<7} | {r['true_position']:<8} | {r['type']:<9} | {det:<9} | {alm:<8} | {dly:<7} |"
    )
print("  +---------+----------+-----------+-----------+----------+---------+")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Remaining plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Generating plots ...")

fig_report, _ = plot_detection_report(
    stream,
    alarm_positions=alarm_positions,
    ise_history=detector.ise_history,
    threshold=ISE_THRESHOLD,
    title="Detection report — caught (green dotted) vs missed",
)
save_fig(fig_report, "02_detection_report.png")

fig3, ax = plt.subplots(figsize=(12, 6))
fig3.suptitle("PDF evolution at key time-steps (Hermite series)", fontsize=13)
palette = [
    "#534AB7",
    "#0F6E56",
    "#D85A30",
    "#185FA5",
    "#993556",
    "#BA7517",
    "#3B6D11",
    "#A32D2D",
]
for (t, pdf), col in zip(sorted(pdf_snapshots.items()), palette):
    lbl = f"n={t}"
    if t == WARMUP:
        lbl += " (reference)"
    for j, spec in enumerate(DRIFT_SPECS):
        if t == spec.position - 1:
            lbl += f" pre-drift {j+1}"
        if t == spec.position + 100:
            lbl += f" post-drift {j+1}"
    if t == N - 1:
        lbl += " (final)"
    ax.plot(NET, pdf, color=col, linewidth=1.4, label=lbl)
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
save_fig(fig3, "03_pdf_evolution.png")


config = {
    "N": N,
    "SEED": SEED,
    "Q_PARAM": Q_PARAM,
    "K_PARAM": K_PARAM,
    "GAMMA": GAMMA,
    "WARMUP": WARMUP,
    "ISE_THRESHOLD": ISE_THRESHOLD,
    "NET_SIZE": len(NET),
    "DRIFTS": [
        {
            "position": d.position,
            "type": d.drift_type,
            "mean_before": d.mean_before,
            "std_before": d.std_before,
            "mean_after": d.mean_after,
            "std_after": d.std_after,
            "label": d.label,
        }
        for d in DRIFT_SPECS
    ],
}

with open(os.path.join(OUT, "experiment_config.json"), "w") as f:
    json.dump(config, f, indent=4)

print(
    f"""
All outputs are in the folder: {OUT}/

  PLOTS (3 files)
    01_stream_overview.png
    02_detection_report.png
    03_pdf_evolution.png

  CSVs (6 files)
    csv_01_stream_raw.csv       index, value, segment_id, segment_label, true_mean, true_std
    csv_02_drift_ground_truth.csv  the 3 injected drifts
    csv_03_ise_score.csv        ise_score, alarm_fired, above_threshold at every step
    csv_04_coefficients.csv     a0, a1 at every step
    csv_05_detection_results.csv   detected yes/no, alarm_position, delay per drift
    csv_06_pdf_snapshots.csv    PDF value at grid points for 8 key moments
"""
)
