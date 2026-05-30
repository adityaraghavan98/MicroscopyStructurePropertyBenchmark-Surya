from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def nlpd(y_true: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    variance = np.maximum(np.asarray(variance), 1e-9)
    residual = np.asarray(y_true) - np.asarray(mean)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + 0.5 * residual**2 / variance))
