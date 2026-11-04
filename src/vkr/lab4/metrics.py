"""Метрики без torch: Macro-F1 (как в ЛР1: по классам, встречающимся в gold) и ECE."""
from __future__ import annotations

import numpy as np


def macro_f1(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fs = []
    for c in np.unique(y_true):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fs.append(2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0)
    return float(np.mean(fs))


def ece(probs: np.ndarray, y_true, n_bins: int = 15) -> float:
    """Expected Calibration Error по уверенности top-1, равноширокие бины."""
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
