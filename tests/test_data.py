"""Test không cần torch: cửa sổ trượt, chống leakage, scaler, metric."""
import numpy as np

from src.data import (fit_scaler, inverse_target, make_windows, raw_history_windows, transform_x,
                      transform_y)
from src.metrics import block_bootstrap_delta_rmse, regression_metrics


def test_window_alignment():
    T, F, L, H = 100, 3, 10, 6
    X = np.arange(T * F, dtype=float).reshape(T, F)
    y = np.arange(T, dtype=float) * 10
    Xw, yw, tgt = make_windows(X, y, L, H)
    assert Xw.shape == (T - L - H + 1, L, F)
    # window 0 = hàng 0..9, target = hàng 0+9+6 = 15
    assert np.array_equal(Xw[0], X[0:L])
    assert tgt[0] == L - 1 + H and yw[0] == y[tgt[0]]
    # bước cuối window luôn nằm H bước trước target
    assert ((tgt - H) == np.arange(len(tgt)) + L - 1).all()
    # window cuối: target là hàng cuối cùng
    assert tgt[-1] == T - 1


def test_history_matches_target_column():
    raw = np.random.default_rng(0).uniform(10, 500, 200)
    L, H = 12, 6
    Xw, yw, tgt = make_windows(raw[:, None], raw, L, H)
    hist = raw_history_windows(raw, L, len(Xw))
    assert np.array_equal(hist, Xw[:, :, 0])
    assert np.array_equal(hist[:, -1], raw[tgt - H])  # persistence


def test_scaler_roundtrip_and_train_only_stats():
    rng = np.random.default_rng(1)
    X = rng.normal(5, 3, (500, 4))
    y = rng.lognormal(4, 0.5, 500)
    for tf in ("none", "log1p"):
        yt = np.log1p(y) if tf == "log1p" else y
        sc = fit_scaler(X[:350], yt[:350], tf)
        Xs, ys = transform_x(X, sc), transform_y(yt, sc)
        assert np.abs(Xs[:350].mean(0)).max() < 1e-9          # chuẩn hoá theo train
        assert np.abs(Xs[350:].mean(0)).max() > 1e-6          # val/test KHÔNG bị ép về 0
        assert np.allclose(inverse_target(ys, sc), y)          # quay về Wh gốc


def test_metrics():
    y = np.array([100., 200., 300.])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0 and m["r2"] == 1
    m = regression_metrics(y, y + 10)
    assert abs(m["rmse"] - 10) < 1e-9 and abs(m["mae"] - 10) < 1e-9


def test_bootstrap_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 300, 600)
    good = y + rng.normal(0, 5, 600)
    bad = y + rng.normal(0, 40, 600)
    r = block_bootstrap_delta_rmse(y, good, bad, block=20, n_boot=300)
    assert r["delta_rmse"] < 0 and r["significant"]


def test_exclude_features():
    import pandas as pd
    import pytest
    from src.data import prepare_arrays

    n = 50
    df = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=n, freq="10min"),
                       "Appliances": np.arange(n) + 10.0, "T1": np.random.rand(n), "T_out": np.random.rand(n),
                       "RH_1": np.random.rand(n), "rv1": np.random.rand(n)})
    cfg = {"data": {"date_col": "date", "target_col": "Appliances", "drop_cols": ["rv1"], "add_time_features": True,
                    "use_target_as_input": True, "target_transform": "none", "exclude_features": []}}
    full = prepare_arrays(df, cfg)
    cfg["data"]["exclude_features"] = ["T1", "T_out"]
    ab = prepare_arrays(df, cfg)
    assert "T1" in full["feature_names"] and "T1" not in ab["feature_names"] and "T_out" not in ab["feature_names"]
    assert "RH_1" in ab["feature_names"] and "Appliances" in ab["feature_names"]
    assert ab["X"].shape[1] == full["X"].shape[1] - 2
    assert np.array_equal(ab["target"], full["target"])  # nhãn không đổi
    cfg["data"]["exclude_features"] = ["khong_ton_tai"]
    with pytest.raises(ValueError):
        prepare_arrays(df, cfg)
    cfg["data"]["exclude_features"] = ["Appliances"]
    with pytest.raises(ValueError):
        prepare_arrays(df, cfg)
