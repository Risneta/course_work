"""Baseline không dùng deep learning: persistence, moving average, daily-naive, Ridge.

Chạy:  python -m src.baselines
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import add_common_args, config_from_args, p
from .data import inverse_target
from .metrics import regression_metrics
from .utils import ensure_dir, load_json, save_json


def load_split(cfg, name):
    z = np.load(p(cfg, "processed_dir", f"{name}.npz"))
    return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    cfg = config_from_args(ap.parse_args(argv))
    out = ensure_dir(p(cfg, "output_dir", "baselines"))
    sc = load_json(p(cfg, "processed_dir", "scaler.json"))
    tr, va, te = (load_split(cfg, s) for s in ("train", "val", "test"))

    def simple_preds(s):
        h = s["hist_raw"]
        return {
            "persistence": h[:, -1],
            "moving_avg_6": h[:, -6:].mean(axis=1),
            "moving_avg_all": h.mean(axis=1),
            "daily_naive": s["daily_raw"],
        }

    results = {"val": {}, "test": {}}
    preds_test = {}
    for split_name, s in (("val", va), ("test", te)):
        for name, pr in simple_preds(s).items():
            ok = np.isfinite(pr)
            results[split_name][name] = regression_metrics(s["y_raw"][ok], pr[ok])
            if split_name == "test":
                preds_test[name] = pr

    # Ridge trên cửa sổ đã flatten, chọn alpha theo val RMSE (Wh)
    Xtr, Xva, Xte = (s["X"].reshape(len(s["X"]), -1) for s in (tr, va, te))
    best = None
    for a in cfg["baselines"]["ridge_alphas"]:
        m = Ridge(alpha=a).fit(Xtr, tr["y"])
        r = regression_metrics(va["y_raw"], inverse_target(m.predict(Xva), sc))["rmse"]
        if best is None or r < best[0]:
            best = (r, a, m)
    _, alpha, model = best
    results["val"]["ridge"] = regression_metrics(va["y_raw"], inverse_target(model.predict(Xva), sc))
    pr_te = inverse_target(model.predict(Xte), sc)
    results["test"]["ridge"] = regression_metrics(te["y_raw"], pr_te)
    results["ridge_alpha"] = alpha
    preds_test["ridge"] = pr_te

    save_json(results, out / "metrics.json")
    np.savez_compressed(out / "test_predictions.npz", y_raw=te["y_raw"], **preds_test)
    table = pd.DataFrame(results["test"]).T[["rmse", "mae", "mape", "smape", "r2", "n"]]
    table.to_csv(out / "summary.csv")
    print("Baseline (test, đơn vị Wh):")
    print(table.round(3).to_string())
    print(f"[OK] đã lưu vào {out}")


if __name__ == "__main__":
    main()
