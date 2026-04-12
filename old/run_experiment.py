"""
Drift Detection Experiment — Orthogonal Series (Hermite System)
===============================================================
Thesis experiment demonstrating:
  1. Ability of the IPNN model to detect concept drift.
  2. Detection delay (number of samples between true drift
     injection and the alarm).

Experiment design
-----------------
Phase 1 (samples 0 .. DRIFT_POINT-1):  X_i ~ N(0, 1)
Phase 2 (samples DRIFT_POINT .. N-1):  X_i ~ N(MEAN_SHIFT, 1)

Because we created the data ourselves, the exact drift location
DRIFT_POINT is known, so the detection delay can be measured precisely.

Results
-------
* Console report: detection delay and alarm index.
* Three plots:
    - ISE score over time with alarm and true-drift markers.
    - PDF evolution: reference vs post-drift snapshots.
    - Coefficients a_{0n} and a_{1n} over time.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Local modules ─────────────────────────────────────────────────────────────
# Make sure series.py, ipnn.py (IPNN class) and drift_detector.py are in the
# same directory (or on the Python path).
from ipnn import IPNN
from drift_detector import DriftDetector

# ══════════════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
N = 2000  # Total number of stream samples
DRIFT_POINT = 1000  # Known drift injection point (0-based index)
MEAN_SHIFT = 2.0  # How far the mean moves after drift

# IPNN hyper-parameters (see Example 1 in the paper)
Q_PARAM = 0.5  # Rate of change in the number of series components
K_PARAM = 3.0  # Constant for the number of components: q(n) = k*(n+1)^Q
GAMMA = 1.0  # Learning-rate exponent: gamma_n = (n+1)^{-gamma}

# Drift detection
WARMUP = 200  # Samples before reference PDF is locked
ISE_THRESHOLD = 0.05  # ISE threshold that triggers the alarm

# Evaluation grid
NET = np.linspace(-5, 8, 300)  # x-axis for PDF evaluation

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Generate synthetic data stream (known drift location)
# ══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(SEED)

phase1 = rng.normal(loc=0.0, scale=1.0, size=DRIFT_POINT)
phase2 = rng.normal(loc=MEAN_SHIFT, scale=1.0, size=N - DRIFT_POINT)
stream = np.concatenate([phase1, phase2])

print(f"Stream length : {N}")
print(f"True drift at : sample {DRIFT_POINT}  (N(0,1) → N({MEAN_SHIFT},1))")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Initialise models
# ══════════════════════════════════════════════════════════════════════════════
model = IPNN(method="series", kernel="Hermite", Q=Q_PARAM, k=K_PARAM, gamma=GAMMA)
detector = DriftDetector(net_of_x=NET, threshold=ISE_THRESHOLD, warmup=WARMUP)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Stream processing loop
# ══════════════════════════════════════════════════════════════════════════════
# We process samples one-by-one so we can call the detector after every step.
# (train_density_recursive processes the entire array at once; here we call
#  update_aj manually to get per-step PDF estimates.)

coeff_history = []  # Track first two coefficients over time
pdf_snapshots = {}  # Save full PDFs at selected time-steps

SNAPSHOT_TIMES = {
    WARMUP,
    DRIFT_POINT - 1,
    DRIFT_POINT + 50,
    DRIFT_POINT + 200,
    N - 1,
}

alarm_raised = False

for n, x_new in enumerate(stream):
    # --- Update IPNN coefficients with the new sample ---
    model.update_aj(x_new, n, 1.0)

    # --- Evaluate current PDF over the grid ---
    q = int(model.q(n))
    current_pdf = np.array(
        [float(np.inner(model.ker(xi, q)[:q], model.a_j[:q, 0])) for xi in NET]
    )

    # --- Store coefficient snapshot ---
    a0 = float(model.a_j[0, 0]) if len(model.a_j) > 0 else 0.0
    a1 = float(model.a_j[1, 0]) if len(model.a_j) > 1 else 0.0
    coeff_history.append((a0, a1))

    # --- Save PDF snapshot ---
    if n in SNAPSHOT_TIMES:
        pdf_snapshots[n] = current_pdf.copy()

    # --- Run drift detector ---
    if not alarm_raised:
        alarm_raised = detector.update(current_pdf, n)

    # Progress indicator (every 10 %)
    if (n + 1) % (N // 10) == 0:
        print(f"  Processed {n+1:5d} / {N} samples …", end="")
        if detector.alarm_index is not None:
            print(f"  *** ALARM at sample {detector.alarm_index} ***")
        else:
            print()

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Report results
# ══════════════════════════════════════════════════════════════════════════════
delay = detector.detection_delay(DRIFT_POINT)

print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)
if detector.alarm_index is not None:
    print(f"  Alarm raised at sample : {detector.alarm_index}")
    print(f"  True drift at sample   : {DRIFT_POINT}")
    print(f"  Detection delay        : {delay} samples")
else:
    print("  No alarm raised — try lowering ISE_THRESHOLD.")
print("=" * 55)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Plots
# ══════════════════════════════════════════════════════════════════════════════
coeff_arr = np.array(coeff_history)  # shape (N, 2)
ise_arr = np.array(detector.ise_history)

fig, axes = plt.subplots(3, 1, figsize=(12, 13))
fig.suptitle(
    f"Orthogonal Series Drift Detection  |  Hermite system  |  "
    f"Drift: N(0,1) → N({MEAN_SHIFT},1) at t={DRIFT_POINT}",
    fontsize=13,
    y=0.98,
)

# ── Plot 1: ISE score ─────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(ise_arr, color="#534AB7", linewidth=1.0, label="ISE score")
ax.axvline(
    DRIFT_POINT,
    color="#D85A30",
    linestyle="--",
    linewidth=1.5,
    label=f"True drift (t={DRIFT_POINT})",
)
if detector.alarm_index is not None:
    ax.axvline(
        detector.alarm_index,
        color="#1D9E75",
        linestyle=":",
        linewidth=2.0,
        label=f"Alarm (t={detector.alarm_index}, delay={delay})",
    )
ax.axhline(
    ISE_THRESHOLD,
    color="#BA7517",
    linestyle="-.",
    linewidth=1.0,
    label=f"Threshold = {ISE_THRESHOLD}",
)
ax.set_title("ISE score over time")
ax.set_xlabel("Stream index n")
ax.set_ylabel("ISE")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Plot 2: PDF snapshots ─────────────────────────────────────────────────────
ax = axes[1]
colors = ["#534AB7", "#0F6E56", "#D85A30", "#185FA5", "#993556"]
for (t, pdf), col in zip(sorted(pdf_snapshots.items()), colors):
    label = f"n={t}"
    if t == WARMUP:
        label += " (reference)"
    elif t == DRIFT_POINT - 1:
        label += " (pre-drift)"
    elif t >= DRIFT_POINT:
        label += f" (post-drift +{t - DRIFT_POINT})"
    ax.plot(NET, pdf, color=col, linewidth=1.4, label=label)

# True PDFs for reference
true_pre = np.exp(-0.5 * NET**2) / np.sqrt(2 * np.pi)
true_post = np.exp(-0.5 * (NET - MEAN_SHIFT) ** 2) / np.sqrt(2 * np.pi)
ax.plot(NET, true_pre, "k--", linewidth=1.0, alpha=0.5, label="True N(0,1)")
ax.plot(NET, true_post, "k:", linewidth=1.0, alpha=0.5, label=f"True N({MEAN_SHIFT},1)")
ax.set_title("Estimated PDF at selected time-steps")
ax.set_xlabel("x")
ax.set_ylabel("Density f̂_n(x)")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)

# ── Plot 3: Coefficient trajectories ─────────────────────────────────────────
ax = axes[2]
ax.plot(coeff_arr[:, 0], color="#534AB7", linewidth=1.0, label="$a_{0n}$")
ax.plot(coeff_arr[:, 1], color="#D85A30", linewidth=1.0, label="$a_{1n}$")
ax.axvline(
    DRIFT_POINT,
    color="#D85A30",
    linestyle="--",
    linewidth=1.5,
    label=f"True drift (t={DRIFT_POINT})",
)
if detector.alarm_index is not None:
    ax.axvline(
        detector.alarm_index,
        color="#1D9E75",
        linestyle=":",
        linewidth=2.0,
        label=f"Alarm (t={detector.alarm_index})",
    )
ax.set_title("Coefficient trajectories  ($a_{0n}$, $a_{1n}$)")
ax.set_xlabel("Stream index n")
ax.set_ylabel("Coefficient value")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("drift_detection_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved → drift_detection_results.png")
