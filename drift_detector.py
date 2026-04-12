"""
Drift Detector using Orthogonal Series Density Estimation
==========================================================
Monitors a data stream by comparing the current estimated PDF
(built by the IPNN model) against a reference PDF captured
during a stable warm-up phase.

Drift is declared when the Integrated Squared Error (ISE)
between the two PDFs exceeds a threshold.

Usage
-----
See run_experiment.py for a complete worked example.
"""

import numpy as np


class DriftDetector:
    """
    Monitors concept drift by tracking the ISE between a
    reference PDF and the current PDF produced by an IPNN model.

    Parameters
    ----------
    net_of_x : ndarray
        Grid of evaluation points shared with the IPNN model.
    threshold : float
        ISE value above which a drift alarm is raised.
    warmup : int
        Number of initial samples used to build the reference PDF.
        The detector is inactive during this period.
    """

    def __init__(
        self, net_of_x: np.ndarray, threshold: float = 0.05, warmup: int = 200
    ):
        self.net_of_x = net_of_x
        self.threshold = threshold
        self.warmup = warmup

        # Reference PDF (set once after warm-up)
        self.reference_pdf: np.ndarray | None = None

        # Recorded history for plotting
        self.ise_history: list[float] = []
        self.alarm_index: int | None = None  # Stream index when alarm fired

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, current_pdf: np.ndarray, stream_index: int) -> bool:
        """
        Evaluate the current PDF against the reference.

        Parameters
        ----------
        current_pdf : ndarray
            Density values at net_of_x, produced by the IPNN model
            after processing stream_index samples.
        stream_index : int
            The current position in the stream (0-based).

        Returns
        -------
        bool
            True if a drift alarm is raised, False otherwise.
        """
        # During warm-up: capture the reference PDF at the last warm-up step
        if stream_index < self.warmup:
            self.ise_history.append(0.0)
            return False

        if stream_index == self.warmup:
            self.reference_pdf = current_pdf.copy()
            self.ise_history.append(0.0)
            return False

        # Compute ISE via trapezoidal integration
        diff = current_pdf - self.reference_pdf
        ise = float(np.trapezoid(diff**2, self.net_of_x))
        self.ise_history.append(ise)

        # Raise alarm on first threshold crossing (only once)
        if self.alarm_index is None and ise > self.threshold:
            self.alarm_index = stream_index
            return True

        return False

    def detection_delay(self, true_drift_point: int) -> int | None:
        """
        Compute how many samples elapsed between the true drift
        point and the detection alarm.

        Parameters
        ----------
        true_drift_point : int
            Stream index at which drift was injected.

        Returns
        -------
        int | None
            Delay in samples, or None if no alarm was raised.
        """
        if self.alarm_index is None:
            return None
        return self.alarm_index - true_drift_point
