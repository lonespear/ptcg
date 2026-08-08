"""D47 numpy-only forward pass for the neural leaf.

No torch at runtime: the export is plain weight matrices in an .npz, and
scoring a leaf is two small matmuls. `score_features` returns the ranking
head; `win_prob` the calibrated-ish value head (sigmoid of the value output).
"""

from __future__ import annotations

import numpy as np


class NeuralLeaf:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.W1 = z["W1"]; self.b1 = z["b1"]
        self.W2 = z["W2"]; self.b2 = z["b2"]
        self.ws = z["ws"]; self.bs = float(z["bs"])
        self.wv = z["wv"]; self.bv = float(z["bv"])
        self.mu = z["mu"]; self.sd = z["sd"]
        self.feature_names = [str(x) for x in z["feature_names"]]
        self.n_features = len(self.feature_names)

    def _trunk(self, x) -> np.ndarray:
        x = (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
        h1 = np.maximum(x @ self.W1 + self.b1, 0)
        return np.maximum(h1 @ self.W2 + self.b2, 0)

    def score_features(self, x) -> float:
        """Ranking-head score for one feature vector (higher = better)."""
        return float(self._trunk(x) @ self.ws + self.bs)

    def win_prob(self, x) -> float:
        v = float(self._trunk(x) @ self.wv + self.bv)
        return 1.0 / (1.0 + np.exp(-v))

    def attributions(self, x, head: str = "score") -> list[tuple[str, float]]:
        """Gradient x input on the named features, for the Strategy report.

        The gradient is exact for the piecewise-linear net: backprop through
        the ReLU masks at x. Returned sorted by |contribution|.
        """
        x = np.asarray(x, dtype=np.float32)
        xs = (x - self.mu) / self.sd
        h1p = xs @ self.W1 + self.b1
        h1 = np.maximum(h1p, 0)
        h2p = h1 @ self.W2 + self.b2
        w = self.ws if head == "score" else self.wv
        d2 = w * (h2p > 0)
        d1 = (self.W2 @ d2) * (h1p > 0)
        dx = (self.W1 @ d1) / self.sd            # d(score)/d(raw feature)
        contrib = dx * (x - self.mu)             # grad x (input - baseline)
        pairs = list(zip(self.feature_names, contrib.tolist()))
        pairs.sort(key=lambda t: -abs(t[1]))
        return pairs
