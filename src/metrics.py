"""Metric hồi quy (đơn vị Wh) + paired moving-block bootstrap. Chỉ dùng numpy."""
import numpy as np


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100),
        "smape": float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def rmse(y_true, y_pred):
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def block_bootstrap_delta_rmse(y_true, pred_a, pred_b, block=36, n_boot=1000, seed=0):
    """Δ = RMSE(a) - RMSE(b). Δ < 0 nghĩa là a tốt hơn b.
    Moving-block bootstrap giữ tương quan thời gian. Trả về mean và CI 95%."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ea = (np.asarray(pred_a, dtype=np.float64) - y_true) ** 2
    eb = (np.asarray(pred_b, dtype=np.float64) - y_true) ** 2
    T = len(y_true)
    block = max(1, min(block, T))
    n_blocks = int(np.ceil(T / block))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        deltas[i] = np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_rmse": float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "significant": bool(hi < 0 or lo > 0),
    }
