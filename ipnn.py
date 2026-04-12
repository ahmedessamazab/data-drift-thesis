"""
IPNN — Incremental Probabilistic Neural Network
================================================
Density estimation in non-stationary environments using
orthonormal series (Hermite, Legendre, Laguerre, etc.).

Reference
---------
Danuta Rutkowska et al., "Probabilistic neural networks for
incremental learning over time-varying streaming data with
application to air pollution monitoring", Applied Soft Computing,
Elsevier, 2024.
"""

import sys
import math
import time

import numpy as np
import series


class IPNN:
    """
    Incremental Probabilistic Neural Network for density estimation.

    Parameters
    ----------
    method : str
        Must be 'series'.
    kernel : str
        Name of the orthogonal basis, e.g. 'Hermite'.
        Must match a key in series.choices.
    Q : float
        Exponent controlling the growth rate of the number of
        series components: q(n) = k * (n+1)^Q.
    k : float
        Scale constant for q(n).
    gamma : float
        Exponent for the learning rate: γ_n = (n+1)^{-gamma}.
    """

    def __init__(self, method: str, kernel: str, Q: float, k: float, gamma: float):
        self.ker = series.choices.get(kernel)
        if self.ker is None:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Available: {list(series.choices)}"
            )

        self.Q = Q
        self.k = k
        self.gamma = gamma

        self.a_j = np.zeros((0, 2))  # columns: [coefficient, update_count]
        self.n = 0
        self.results = np.zeros(0)
        self.l_time = 0.0

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __str__(self):
        return "IPNN model"

    def __repr__(self):
        name = self.ker.__name__
        return f"IPNN(kernel={name}, Q={self.Q}, k={self.k}, γ={self.gamma})"

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def q(self, i: int) -> float:
        """Number of series components at step i: k * (i+1)^Q."""
        return self.k * math.pow(i + 1, self.Q)

    def gamma_n(self, i: int) -> float:
        """Learning rate at step i: (i+1)^{-gamma}."""
        return math.pow(i + 1, -self.gamma)

    # ------------------------------------------------------------------
    # Core recursive update  (Eq. 13 in the paper)
    # ------------------------------------------------------------------

    def update_aj(self, x: float, i: int, y: float = 1.0) -> None:
        """
        Update model coefficients a_{jn} for a new sample x at step i.

        Parameters
        ----------
        x : float
            New incoming data point.
        i : int
            Current stream index (0-based).
        y : float
            Target value (always 1.0 for density estimation).
        """
        q = int(self.q(i))
        current_element = self.ker(x, q)

        # Expand coefficient array if needed
        shortfall = q - len(self.a_j)
        if shortfall > 0:
            self.a_j = np.append(self.a_j, np.zeros((shortfall, 2)), axis=0)

        lr = self.gamma_n(i)
        for d in range(len(self.a_j)):
            if self.a_j[d, 1] == 0:
                # First update for this component
                self.a_j[d, 1] = 1
                self.a_j[d, 0] = y * current_element[d]
            else:
                self.a_j[d, 0] *= 1.0 - lr
                self.a_j[d, 0] += y * current_element[d] * lr
                self.a_j[d, 1] += 1

    # ------------------------------------------------------------------
    # Batch training (processes entire dataset, stores all timesteps)
    # ------------------------------------------------------------------

    def train_density_recursive(
        self,
        data: np.ndarray,
        net_of_x: np.ndarray,
    ) -> np.ndarray:
        """
        Process all samples in *data* and store the estimated PDF at
        every timestep.

        Parameters
        ----------
        data : ndarray, shape (N,)
            Input data stream.
        net_of_x : ndarray, shape (M,)
            Evaluation grid for the PDF.

        Returns
        -------
        ndarray, shape (M,)
            Final PDF estimate (last timestep).
        """
        start = time.time()
        N = len(data)
        M = len(net_of_x)
        result_x = np.zeros((M, N))

        for j in range(N):
            self.update_aj(data[j], j, 1.0)
            q = int(self.q(j))
            for xi in range(M):
                net_part = self.ker(net_of_x[xi], q)
                result_x[xi, j] = float(np.inner(net_part[:q], self.a_j[:q, 0]))
            sys.stdout.write(f"\r Processed: {100 * j / N:.0f}%")
            sys.stdout.flush()

        self.results = result_x
        self.l_time = time.time() - start
        return result_x[:, -1]

    # ------------------------------------------------------------------
    # Sliding-window KDE (stationary baseline)
    # ------------------------------------------------------------------

    def train_density_windows(
        self,
        data: np.ndarray,
        net_of_x: np.ndarray,
        window_size: int = 200,
        end: int = 0,
    ) -> np.ndarray:
        """
        Kernel density estimation with a sliding window.

        Parameters
        ----------
        data : ndarray
            Full data stream.
        net_of_x : ndarray
            Evaluation grid.
        window_size : int
            Number of samples in each window.
        end : int
            Last sample index of the window (0 → use final window).

        Returns
        -------
        ndarray
            PDF estimate over net_of_x for the chosen window.
        """
        if end == 0 or end == len(data):
            X = data[-window_size:]
        else:
            X = data[end - window_size : end + 1]

        start = time.time()
        M = len(net_of_x)
        result_x = np.zeros(M)

        self._update_aj_window(X, window_size)
        q = int(self.q(window_size))

        for xi in range(M):
            net_part = self.ker(net_of_x[xi], q)
            result_x[xi] = float(np.inner(net_part[:q], self.a_j[:q, 0]))

        self.results = result_x
        self.l_time = time.time() - start
        return result_x

    def _update_aj_window(self, x: np.ndarray, i: int) -> None:
        """Internal helper: compute coefficients for a fixed window."""
        q = int(self.q(i))
        self.a_j = np.zeros((q, 2))

        for j, xj in enumerate(x):
            current_element = self.ker(xj, q)
            for d in range(q):
                self.a_j[d, 0] += current_element[d] / i
                self.a_j[d, 1] += 1
            sys.stdout.write(f"\r Processed: {100 * j / i:.0f}%")
            sys.stdout.flush()
