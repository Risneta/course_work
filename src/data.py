"""Đọc dữ liệu, kiểm tra, tạo feature, scaler và cửa sổ trượt. Chỉ dùng numpy/pandas."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


# --------------------------------------------------------------------- load / validate
def load_raw(path, date_col="date"):
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Không thấy cột '{date_col}' trong {path}. Cột hiện có: {list(df.columns)[:8]}...")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def validate_raw(df, cfg):
    """Trả về report dict. Raise ValueError nếu lỗi nghiêm trọng (thiếu cột target/date, cột không phải số)."""
    d = cfg["data"]
    if d["target_col"] not in df.columns:
        raise ValueError(f"Thiếu cột target '{d['target_col']}'")
    non_numeric = [c for c in df.columns if c != d["date_col"] and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Cột không phải số: {non_numeric}")

    diffs = df[d["date_col"]].diff().dropna()
    expected = pd.Timedelta(minutes=d["freq_minutes"])
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "expected_rows": d.get("expected_rows"),
        "rows_match_expected": bool(len(df) == d.get("expected_rows")),
        "date_min": str(df[d["date_col"]].min()),
        "date_max": str(df[d["date_col"]].max()),
        "n_missing_cells": int(df.isna().sum().sum()),
        "n_duplicate_timestamps": int(df[d["date_col"]].duplicated().sum()),
        "n_irregular_steps": int((diffs != expected).sum()),
        "max_gap_minutes": float(diffs.max().total_seconds() / 60) if len(diffs) else 0.0,
        "warnings": [],
    }
    if not report["rows_match_expected"]:
        report["warnings"].append(f"Số dòng {len(df)} khác expected_rows {d.get('expected_rows')}")
    if report["n_missing_cells"]:
        report["warnings"].append("Có giá trị thiếu: sẽ nội suy tuyến tính")
    if report["n_duplicate_timestamps"]:
        report["warnings"].append("Có timestamp trùng")
    if report["n_irregular_steps"]:
        report["warnings"].append("Có bước thời gian không đều (khác freq_minutes)")
    return report


# --------------------------------------------------------------------- features
def prepare_arrays(df, cfg):
    d = cfg["data"]
    tgt = d["target_col"]
    dates = df[d["date_col"]]
    raw_target = df[tgt].to_numpy(dtype=np.float64)

    work = df.drop(columns=[d["date_col"]] + [c for c in d["drop_cols"] if c in df.columns]).copy()
    work = work.interpolate(limit_direction="both")
    if d.get("target_transform", "none") == "log1p":
        work[tgt] = np.log1p(work[tgt])

    if d["add_time_features"]:
        hod = dates.dt.hour + dates.dt.minute / 60.0
        dow = dates.dt.dayofweek
        work["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
        work["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
        work["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        work["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        work["is_weekend"] = (dow >= 5).astype(float)

    target = work[tgt].to_numpy(dtype=np.float64)
    feat_cols = list(work.columns) if d["use_target_as_input"] else [c for c in work.columns if c != tgt]

    # Ablation (D21): loại một số feature khỏi INPUT. Target không bao giờ bị loại khỏi việc tạo nhãn.
    excl = list(d.get("exclude_features") or [])
    if excl:
        if tgt in excl:
            raise ValueError("exclude_features không được chứa cột target; dùng data.use_target_as_input=false")
        unknown = [c for c in excl if c not in work.columns]
        if unknown:
            raise ValueError(f"exclude_features chứa cột không tồn tại: {unknown}. Cột có: {list(work.columns)}")
        feat_cols = [c for c in feat_cols if c not in excl]
        if not feat_cols:
            raise ValueError("Không còn feature nào sau khi loại exclude_features")
    return {
        "X": work[feat_cols].to_numpy(dtype=np.float64),
        "feature_names": feat_cols,
        "target": target,          # sau transform (nếu có), CHƯA scale
        "raw_target": raw_target,  # Wh gốc
        "dates": dates.to_numpy().astype("datetime64[s]"),
    }


# --------------------------------------------------------------------- split / scaler
def split_bounds(n_rows, cfg):
    """Biên theo chỉ số hàng: train = [0, train_end), val = [train_end, val_end), test = [val_end, n_rows)."""
    s = cfg["split"]
    train_end = int(n_rows * s["train"])
    val_end = int(n_rows * (s["train"] + s["val"]))
    return train_end, val_end


def fit_scaler(X_train, y_train, target_transform="none"):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return {
        "x_mean": mu.tolist(),
        "x_std": sd.tolist(),
        "y_mean": float(y_train.mean()),
        "y_std": float(max(y_train.std(), 1e-8)),
        "target_transform": target_transform,
    }


def transform_x(X, sc):
    return (X - np.asarray(sc["x_mean"])) / np.asarray(sc["x_std"])


def transform_y(y, sc):
    return (y - sc["y_mean"]) / sc["y_std"]


def inverse_target(y_scaled, sc):
    """Từ target đã scale về Wh gốc."""
    y = np.asarray(y_scaled, dtype=np.float64) * sc["y_std"] + sc["y_mean"]
    if sc.get("target_transform", "none") == "log1p":
        y = np.expm1(y)
    return y


# --------------------------------------------------------------------- windows
def make_windows(X_all, y_all, lookback, horizon):
    """Window i dùng hàng [i, i+L-1]; target là y_all[i+L-1+H].
    Trả về X (N,L,F), y (N,), tgt_idx (N,) = chỉ số hàng của target."""
    T = X_all.shape[0]
    n = T - lookback - horizon + 1
    if n <= 0:
        raise ValueError("Dữ liệu quá ngắn so với lookback + horizon")
    W = sliding_window_view(X_all, window_shape=lookback, axis=0)  # (T-L+1, F, L)
    X = np.ascontiguousarray(W[:n].transpose(0, 2, 1))
    tgt_idx = np.arange(n) + lookback - 1 + horizon
    return X, y_all[tgt_idx], tgt_idx


def raw_history_windows(raw_target, lookback, n):
    """(N, L) lịch sử Appliances gốc (Wh) của từng window, dùng cho baseline."""
    W = sliding_window_view(raw_target, window_shape=lookback)  # (T-L+1, L)
    return np.ascontiguousarray(W[:n])
