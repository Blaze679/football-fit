"""Small statistics helpers for the Backtest page (numpy/pandas only, seeded, no extra dependencies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney) formula; ties get average ranks."""
    y = np.asarray(y_true).astype(int)
    r = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(np.asarray(x, dtype=float)).rank().corr(pd.Series(np.asarray(y, dtype=float)).rank()))


def bootstrap_ci(func, arrays: tuple[np.ndarray, ...], n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Point estimate and 95% percentile bootstrap interval (resampling players)."""
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    est = float(func(*arrays))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = func(*(a[idx] for a in arrays))
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < 50:
        return est, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, float(lo), float(hi)


def perm_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 4000, seed: int = 7) -> float:
    """Two-sided permutation p-value for the Spearman correlation."""
    rx = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, dtype=float)).rank().to_numpy()
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= obs for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))
