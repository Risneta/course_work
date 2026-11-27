"""Bước preprocess: raw CSV -> train/val/test .npz (đã scale, đã tạo window).

Chạy:  python -m src.preprocess
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import add_common_args, config_from_args, p
from .data import (fit_scaler, load_raw, make_windows, prepare_arrays, raw_history_windows,
                   split_bounds, transform_x, transform_y, validate_raw)
from .utils import ensure_dir, save_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    cfg = config_from_args(ap.parse_args(argv))

    d, L, H = cfg["data"], cfg["window"]["lookback"], cfg["window"]["horizon"]
    out_dir = ensure_dir(p(cfg, "processed_dir"))

    # 1. đọc + kiểm tra
    df = load_raw(p(cfg, "raw_csv"), d["date_col"])
    report = validate_raw(df, cfg)
    for w in report["warnings"]:
        print("[WARN]", w)

    # 2. feature + target
    arr = prepare_arrays(df, cfg)
    X, target, raw_target = arr["X"], arr["target"], arr["raw_target"]
    T = len(X)

    # 3. biên chia theo thời gian + fit scaler CHỈ trên train
    train_end, val_end = split_bounds(T, cfg)
    sc = fit_scaler(X[:train_end], target[:train_end], d.get("target_transform", "none"))
    Xs, ys = transform_x(X, sc), transform_y(target, sc)

    # 4. tạo window trên toàn chuỗi, gán split theo chỉ số của bước TARGET
    Xw, yw, tgt_idx = make_windows(Xs, ys, L, H)
    hist_raw = raw_history_windows(raw_target, L, len(Xw))
    y_raw = raw_target[tgt_idx]
    daily_lag = cfg["baselines"]["daily_lag_steps"]
    daily_idx = tgt_idx - daily_lag
    daily_raw = np.where(daily_idx >= 0, raw_target[np.clip(daily_idx, 0, None)], np.nan)
    tgt_time = arr["dates"][tgt_idx]

    masks = {
        "train": tgt_idx < train_end,
        "val": (tgt_idx >= train_end) & (tgt_idx < val_end),
        "test": tgt_idx >= val_end,
    }

    # 5. kiểm tra chống leakage
    assert np.isfinite(Xw).all() and np.isfinite(yw).all(), "Có NaN/inf sau preprocess"
    assert tgt_idx[masks["train"]].max() < train_end <= tgt_idx[masks["val"]].min()
    assert tgt_idx[masks["val"]].max() < val_end <= tgt_idx[masks["test"]].min()
    train_rows = Xs[:train_end]
    assert np.abs(train_rows.mean(axis=0)).max() < 1e-6, "Scaler không khớp thống kê train"
    # target của window = hàng tgt_idx; lịch sử input kết thúc ở tgt_idx - H  (luôn là quá khứ so với target)
    assert (tgt_idx - H == np.arange(len(tgt_idx)) + L - 1).all()

    # 6. lưu
    split_info = {"n_rows": T, "lookback": L, "horizon": H, "train_end_row": train_end, "val_end_row": val_end,
                  "train_period": [str(arr["dates"][0]), str(arr["dates"][train_end - 1])],
                  "val_period": [str(arr["dates"][train_end]), str(arr["dates"][val_end - 1])],
                  "test_period": [str(arr["dates"][val_end]), str(arr["dates"][-1])],
                  "n_windows": {k: int(m.sum()) for k, m in masks.items()}}
    for name, m in masks.items():
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X=Xw[m].astype(np.float32), y=yw[m].astype(np.float32),
            y_raw=y_raw[m].astype(np.float32), hist_raw=hist_raw[m].astype(np.float32),
            daily_raw=daily_raw[m].astype(np.float32), tgt_idx=tgt_idx[m].astype(np.int32),
            tgt_time=tgt_time[m])
    save_json(sc, out_dir / "scaler.json")
    save_json(arr["feature_names"], out_dir / "feature_names.json")
    save_json(split_info, out_dir / "split_info.json")
    save_json({"validation": report, "n_features": len(arr["feature_names"]),
               "feature_names": arr["feature_names"], "split": split_info},
              out_dir / "preprocess_report.json")

    # 7. hình kiểm tra split
    fig, ax = plt.subplots(figsize=(12, 3))
    t = arr["dates"]
    for (a, b), c, lab in [((0, train_end), "#1f77b4", "train"), ((train_end, val_end), "#ff7f0e", "val"),
                           ((val_end, T), "#2ca02c", "test")]:
        ax.plot(t[a:b], raw_target[a:b], lw=0.4, color=c, label=lab)
    ax.set_ylabel("Appliances (Wh)")
    ax.set_title("Chia train / val / test theo thời gian")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "split_timeline.png", dpi=130)
    plt.close(fig)

    print(f"[OK] windows: {split_info['n_windows']} | features: {len(arr['feature_names'])} | L={L}, H={H}")
    print(f"[OK] đã lưu vào {out_dir}")


if __name__ == "__main__":
    main()
