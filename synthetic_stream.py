"""
Enhanced Synthetic Data Generator with Multiple Drift Points
=============================================================
Creates a data stream with MULTIPLE known drift points so you can:
  1. See the stream visually before running detection.
  2. Know exactly where drifts are injected.
  3. After detection, see which drifts were caught and which were missed.

Drift types supported
---------------------
  - 'mean'         : sudden shift in distribution mean
  - 'variance'     : sudden change in standard deviation
  - 'gradual'      : mean and/or std drift smoothly over a transition window
  - 'cyclic'       : periodic (sinusoidal) change around the current mean
  - 'distribution' : structural change in distribution (e.g., Gaussian → Uniform)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DriftSpec:
    """Specification for a single drift event injected into the stream."""

    position: int  # Stream index where drift begins
    drift_type: str  # 'mean', 'variance', 'gradual'
    mean_before: float
    std_before: float
    mean_after: float
    std_after: float
    transition_width: int = 0  # Only used for 'gradual' drift
    label: str = ""

    def __post_init__(self):
        if not self.label:
            delta = self.mean_after - self.mean_before
            self.label = (
                f"{self.drift_type.capitalize()} drift "
                f"(μ: {self.mean_before}→{self.mean_after}, "
                f"σ: {self.std_before}→{self.std_after})"
            )


@dataclass
class SyntheticStream:
    """Container for a generated stream and its ground-truth drift map."""

    data: np.ndarray
    drift_specs: list
    segment_labels: list  # Human-readable label for each sample's segment
    true_drift_positions: list  # Sorted list of all true drift indices


# ══════════════════════════════════════════════════════════════════════════════
# Generator
# ══════════════════════════════════════════════════════════════════════════════


class SyntheticStreamGenerator:
    """
    Builds a data stream from a sequence of DriftSpec objects.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, total_length: int, drift_specs: list) -> SyntheticStream:
        """
        Generate a stream of *total_length* samples with the given drifts.

        Parameters
        ----------
        total_length : int
            Total number of samples.
        drift_specs : list of DriftSpec
            Drift events, sorted by position.

        Returns
        -------
        SyntheticStream
        """
        drift_specs = sorted(drift_specs, key=lambda d: d.position)
        data = np.zeros(total_length)
        segment_labels = [""] * total_length

        # Build segment boundaries
        boundaries = [d.position for d in drift_specs] + [total_length]
        starts = [0] + [d.position for d in drift_specs]

        current_mean = drift_specs[0].mean_before if drift_specs else 0.0
        current_std = drift_specs[0].std_before if drift_specs else 1.0

        for seg_idx, (seg_start, seg_end) in enumerate(zip(starts, boundaries)):
            if seg_idx > 0:
                spec = drift_specs[seg_idx - 1]
                current_mean = spec.mean_after
                current_std = spec.std_after

            for i in range(seg_start, seg_end):
                if seg_idx > 0:
                    spec = drift_specs[seg_idx - 1]
                    drift_type = spec.drift_type

                    # GRADUAL
                    if drift_type == "gradual":
                        tw = spec.transition_width
                        progress = (i - seg_start) / max(tw, 1)
                        progress = np.clip(progress, 0, 1)

                        m = spec.mean_before + progress * (
                            spec.mean_after - spec.mean_before
                        )
                        s = spec.std_before + progress * (
                            spec.std_after - spec.std_before
                        )

                        data[i] = self.rng.normal(m, s)

                    # CYCLIC
                    elif drift_type == "cyclic":
                        t = i - seg_start
                        base_mean = current_mean
                        amplitude = 2.0
                        frequency = 50
                        m = base_mean + amplitude * np.sin(t / frequency)
                        s = current_std
                        data[i] = self.rng.normal(m, s)

                    # DISTRIBUTION CHANGE
                    elif drift_type == "distribution":
                        low = current_mean - 3 * current_std
                        high = current_mean + 3 * current_std
                        data[i] = self.rng.uniform(low, high)

                    # MEAN / VARIANCE (DEFAULT)
                    else:
                        data[i] = self.rng.normal(current_mean, current_std)

                else:
                    data[i] = self.rng.normal(current_mean, current_std)

                # Label
                if seg_idx == 0:
                    segment_labels[i] = f"Stable (μ={current_mean}, σ={current_std})"
                else:
                    spec = drift_specs[seg_idx - 1]
                    segment_labels[i] = spec.label

        return SyntheticStream(
            data=data,
            drift_specs=drift_specs,
            segment_labels=segment_labels,
            true_drift_positions=[d.position for d in drift_specs],
        )


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

DRIFT_COLORS = ["#D85A30", "#534AB7", "#1D9E75", "#BA7517", "#993556"]


def plot_stream(
    stream: SyntheticStream, window: int = 50, title: str = "Synthetic data stream"
) -> plt.Figure:
    """
    Plot the raw stream and its rolling statistics, with drift markers.

    Parameters
    ----------
    stream : SyntheticStream
    window : int
        Rolling mean/std window size.
    title : str
    """
    data = stream.data
    N = len(data)
    roll_mean = np.convolve(data, np.ones(window) / window, mode="same")
    roll_std = np.array(
        [data[max(0, i - window // 2) : i + window // 2].std() for i in range(N)]
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13, y=0.98)

    # ── Raw stream ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(range(N), data, s=1.5, color="#888780", alpha=0.5, label="samples")
    ax.plot(
        roll_mean, color="#534AB7", linewidth=1.2, label=f"rolling mean (w={window})"
    )
    ax.set_ylabel("Value")
    ax.set_title("Raw stream + rolling mean")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    # ── Rolling std ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(roll_std, color="#D85A30", linewidth=1.0, label=f"rolling std (w={window})")
    ax.set_ylabel("Std")
    ax.set_title("Rolling standard deviation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # ── Segment labels ──────────────────────────────────────────────────────
    ax = axes[2]
    # Shade each segment a different colour
    boundaries = [0] + stream.true_drift_positions + [N]
    seg_colors = ["#E6F1FB", "#EAF3DE", "#FAEEDA", "#FAECE7", "#FBEAF0", "#E1F5EE"]
    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(a, b, color=seg_colors[i % len(seg_colors)], alpha=0.6)

    ax.plot(data, color="#888780", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Value")
    ax.set_title("Segment map (shaded regions)")
    ax.set_xlabel("Stream index n")
    ax.grid(alpha=0.2)

    # ── Drift markers on all subplots ────────────────────────────────────────
    for spec, col in zip(stream.drift_specs, DRIFT_COLORS):
        for ax in axes:
            ax.axvline(
                spec.position, color=col, linestyle="--", linewidth=1.5, alpha=0.8
            )
        # Annotation only on top subplot
        axes[0].annotate(
            f"Drift {stream.true_drift_positions.index(spec.position)+1}\n"
            f"t={spec.position}",
            xy=(spec.position, axes[0].get_ylim()[1] * 0.95),
            xytext=(spec.position + N * 0.01, axes[0].get_ylim()[1] * 0.9),
            fontsize=8,
            color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_detection_report(
    stream: SyntheticStream,
    alarm_positions: list,
    ise_history: list | np.ndarray,
    threshold: float,
    title: str = "Detection report",
) -> plt.Figure:
    """
    After running the detector, show which drifts were caught, which were
    missed, and the ISE score trajectory.

    Parameters
    ----------
    stream : SyntheticStream
    alarm_positions : list of int
        Stream indices where alarms were raised (can be multiple).
    ise_history : array-like
        ISE score at every stream step.
    threshold : float
        ISE threshold used.
    """
    ise = np.asarray(ise_history)
    N = len(ise)
    true_positions = stream.true_drift_positions

    # Match alarms to drifts: each alarm is attributed to the nearest
    # preceding true drift that has not been claimed yet.
    matched = {}  # drift_index → alarm_index
    unmatched_alarms = []
    claimed = set()

    for alarm in sorted(alarm_positions):
        # Find the most recent true drift that precedes this alarm
        candidates = [
            (i, p)
            for i, p in enumerate(true_positions)
            if p <= alarm and i not in claimed
        ]
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            matched[best[0]] = alarm
            claimed.add(best[0])
        else:
            unmatched_alarms.append(alarm)

    # Build report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  DRIFT DETECTION REPORT")
    report_lines.append("=" * 60)
    for i, spec in enumerate(stream.drift_specs):
        if i in matched:
            delay = matched[i] - spec.position
            status = f"✓ DETECTED   alarm at t={matched[i]}  delay={delay} samples"
            col = "#1D9E75"
        else:
            status = "✗ MISSED"
            col = "#D85A30"
        report_lines.append(
            f"  Drift {i+1} at t={spec.position}  [{spec.drift_type}]  {status}"
        )
    if unmatched_alarms:
        report_lines.append(f"\n  False alarms at: {unmatched_alarms}")
    report_lines.append("=" * 60)
    full_report = "\n".join(report_lines)
    print(full_report)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=13, y=0.98)

    # ISE score
    ax1.plot(ise, color="#534AB7", linewidth=1.0, label="ISE score")
    ax1.axhline(
        threshold,
        color="#BA7517",
        linestyle="-.",
        linewidth=1.2,
        label=f"Threshold = {threshold}",
    )

    # True drift lines
    for i, (spec, col) in enumerate(zip(stream.drift_specs, DRIFT_COLORS)):
        ax1.axvline(
            spec.position,
            color=col,
            linestyle="--",
            linewidth=1.5,
            label=f"True drift {i+1} (t={spec.position})",
        )

    # Alarm lines
    for alarm in sorted(alarm_positions):
        ax1.axvline(alarm, color="#1D9E75", linestyle=":", linewidth=2.0, alpha=0.9)

    ax1.set_ylabel("ISE")
    ax1.set_title("ISE score over time")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(alpha=0.25)

    # Detection status panel
    ax2.plot(stream.data, color="#888780", linewidth=0.5, alpha=0.4)
    for i, spec in enumerate(stream.drift_specs):
        col = DRIFT_COLORS[i % len(DRIFT_COLORS)]
        ax2.axvline(spec.position, color=col, linestyle="--", linewidth=1.5)
        status_text = (
            f"✓ detected\ndelay={matched[i]-spec.position}"
            if i in matched
            else "✗ missed"
        )
        ax2.text(
            spec.position + N * 0.005,
            ax2.get_ylim()[1] * 0.85 - i * 0.4,
            f"Drift {i+1}\n{status_text}",
            fontsize=8,
            color=col,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.8),
        )

    for alarm in sorted(alarm_positions):
        ax2.axvline(alarm, color="#1D9E75", linestyle=":", linewidth=2.0, alpha=0.9)

    # Legend patches
    legend_elements = [
        mpatches.Patch(color="#D85A30", label="True drift"),
        mpatches.Patch(color="#1D9E75", label="Alarm fired"),
    ]
    ax2.legend(handles=legend_elements, fontsize=9)
    ax2.set_xlabel("Stream index n")
    ax2.set_ylabel("Value")
    ax2.set_title("Stream with detection overlay")
    ax2.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, full_report
