"""Đánh giá trên tập test (đơn vị Wh), so sánh Transformer vs LSTM vs baseline, kiểm tra tiêu chí đạt/không đạt.

Chạy:  python -m src.evaluate
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import add_common_args, config_from_args, p
from .data import inverse_target
from .metrics import block_bootstrap_delta_rmse, regression_metrics
from .models import load_checkpoint
from .train import get_device, load_split, predict
from .utils import ensure_dir, load_json, save_json

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
COLORS = {"transformer": "#d62728", "lstm": "#1f77b4", "persistence": "#7f7f7f", "ridge": "#2ca02c"}
METRICS = ["rmse", "mae", "mape", "smape", "r2"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    cfg = config_from_args(ap.parse_args(argv))
    seeds = cfg["train"]["seeds"]
    device = get_device(cfg)
    sc = load_json(p(cfg, "processed_dir", "scaler.json"))
    test, val = load_split(cfg, "test"), load_split(cfg, "val")
    y = test["y_raw"].astype(np.float64)
    out_metrics = ensure_dir(p(cfg, "output_dir", "metrics"))
    out_pred = ensure_dir(p(cfg, "output_dir", "predictions"))
    out_fig = ensure_dir(p(cfg, "output_dir", "figures"))

    # ------------------------------------------------------------------ 1. dự đoán của từng model/seed
    preds, per_seed, params = {}, [], {}
    for name in ("transformer", "lstm"):
        preds[name] = {}
        for s in seeds:
            ck_path = p(cfg, "output_dir", "models", name, f"seed{s}", "best.pt")
            if not ck_path.exists():
                print(f"[WARN] thiếu checkpoint {ck_path}")
                continue
            model, ck = load_checkpoint(ck_path, device)
            pr = inverse_target(predict(model, test["X"], device), sc)
            pv = inverse_target(predict(model, val["X"], device), sc)
            m_te, m_va = regression_metrics(y, pr), regression_metrics(val["y_raw"], pv)
            per_seed.append({"model": name, "seed": s, "split": "test", **m_te})
            per_seed.append({"model": name, "seed": s, "split": "val", **m_va})
            preds[name][s] = pr
            np.savez_compressed(out_pred / f"{name}_seed{s}_test.npz", y_raw=y, pred=pr, tgt_time=test["tgt_time"])
            params[name] = load_json(p(cfg, "output_dir", "models", name, f"seed{s}", "train_summary.json"))["n_params"]
    if not any(preds[n] for n in preds):
        raise SystemExit("Chưa có checkpoint nào. Chạy `python -m src.train` trước.")
    ps = pd.DataFrame(per_seed)
    ps.to_csv(out_metrics / "per_seed_metrics.csv", index=False)

    # ------------------------------------------------------------------ 2. bảng tổng hợp
    base = np.load(p(cfg, "output_dir", "baselines", "test_predictions.npz"))
    rows = []
    for b in ("persistence", "moving_avg_6", "ridge"):
        if b in base.files:
            rows.append({"model": b, "kind": "baseline", **{f"{k}_mean": v for k, v in
                         regression_metrics(y, base[b]).items() if k in METRICS},
                         **{f"{k}_std": 0.0 for k in METRICS}, "n_seeds": 1, "n_params": 0})
    ens = {}
    for name in ("transformer", "lstm"):
        if not preds[name]:
            continue
        sub = ps[(ps.model == name) & (ps.split == "test")]
        ens[name] = np.mean(np.stack(list(preds[name].values())), axis=0)
        row = {"model": name, "kind": "deep", "n_seeds": len(sub), "n_params": params.get(name, 0)}
        for k in METRICS:
            row[f"{k}_mean"], row[f"{k}_std"] = float(sub[k].mean()), float(sub[k].std(ddof=0))
        rows.append(row)
        e = regression_metrics(y, ens[name])
        rows.append({"model": f"{name}_seed_ensemble", "kind": "deep_ensemble", "n_seeds": len(sub),
                     "n_params": params.get(name, 0), **{f"{k}_mean": e[k] for k in METRICS},
                     **{f"{k}_std": 0.0 for k in METRICS}})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_metrics / "summary.csv", index=False)

    # ------------------------------------------------------------------ 3. so sánh có ý nghĩa thống kê (block bootstrap)
    ec = cfg["eval"]
    comps = {}

    def cmp(a_name, a_pred, b_name, b_pred):
        comps[f"{a_name}_vs_{b_name}"] = block_bootstrap_delta_rmse(y, a_pred, b_pred, ec["bootstrap_block"],
                                                                   ec["bootstrap_n"])

    if "transformer" in ens and "lstm" in ens:
        cmp("transformer", ens["transformer"], "lstm", ens["lstm"])
    for name in ens:
        for b in ("persistence", "moving_avg_6", "ridge"):
            if b in base.files:
                cmp(f"{name}_ensemble", ens[name], b, base[b])
        if "ridge" in base.files:  # từng seed riêng lẻ (công bằng hơn ensemble vs 1 baseline)
            for s, pr in preds[name].items():
                cmp(f"{name}_seed{s}", pr, "ridge", base["ridge"])
    save_json(comps, out_metrics / "comparison_bootstrap.json")

    # ------------------------------------------------------------------ 4. kiểm tra tiêu chí thành công
    def rmse_of(model):
        r = summary[summary.model == model]
        return float(r["rmse_mean"].iloc[0]) if len(r) else None

    checks = []
    for name in ("transformer", "lstm"):
        r = rmse_of(name)
        if r is None:
            continue
        sub_n = ps[(ps.model == name) & (ps.split == "test")]
        for b in ("persistence", "moving_avg_6", "ridge"):
            rb = rmse_of(b)
            if rb is None:
                continue
            tag = " (BẮT BUỘC)" if b == "ridge" else ""
            n_win = int((sub_n["rmse"] < rb).sum())
            checks.append({"id": f"{name}_beats_{b}{tag}: điểm ước lượng (mean qua seed)",
                           "pass": bool(r < rb),
                           "detail": f"RMSE {r:.2f} vs {rb:.2f} Wh; {n_win}/{len(sub_n)} seed thắng"})
            c = comps.get(f"{name}_ensemble_vs_{b}")
            if c:
                checks.append({"id": f"{name}_beats_{b}{tag}: có ý nghĩa thống kê (ensemble, block bootstrap)",
                               "pass": bool(c["ci95_high"] < 0),
                               "detail": f"ΔRMSE={c['delta_rmse']:.2f}, CI95=[{c['ci95_low']:.2f}, {c['ci95_high']:.2f}]"})
    sub_t = ps[(ps.model == "transformer") & (ps.split == "test")]
    sub_l = ps[(ps.model == "lstm") & (ps.split == "test")]
    for name, sub in (("transformer", sub_t), ("lstm", sub_l)):
        if len(sub) > 1:
            cv = sub["rmse"].std(ddof=0) / sub["rmse"].mean()
            checks.append({"id": f"{name}_seed_stability (CV RMSE < 10%)", "pass": bool(cv < 0.10),
                           "detail": f"CV={cv * 100:.1f}% qua {len(sub)} seed"})
    for name in ens:
        vt = ps[(ps.model == name) & (ps.split == "val")]["rmse"].mean()
        tt = ps[(ps.model == name) & (ps.split == "test")]["rmse"].mean()
        checks.append({"id": f"{name}_no_severe_overfit_to_val (test RMSE <= 1.5x val RMSE)",
                       "pass": bool(tt <= 1.5 * vt), "detail": f"val {vt:.2f} / test {tt:.2f} Wh"})
    save_json(checks, out_metrics / "success_checks.json")

    # ------------------------------------------------------------------ 5. hình
    k = min(ec["plot_points"], len(y))
    tt = test["tgt_time"][:k]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(tt, y[:k], color="black", lw=1.2, label="actual")
    for name in ens:
        ax.plot(tt, ens[name][:k], color=COLORS[name], lw=1, label=f"{name} (ensemble seed)")
    ax.plot(tt, base["persistence"][:k], color=COLORS["persistence"], lw=0.8, ls="--", label="persistence")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_title(f"Dự đoán t+{cfg['window']['horizon']} bước trên {k} điểm test đầu tiên")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "pred_vs_actual.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ens), figsize=(5 * len(ens), 4.5), squeeze=False)
    lim = float(max(y.max(), *(e.max() for e in ens.values())))
    for ax, name in zip(axes[0], ens):
        ax.scatter(y, ens[name], s=4, alpha=0.3, color=COLORS[name])
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("actual (Wh)")
        ax.set_ylabel("predicted (Wh)")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_fig / "scatter_pred_vs_actual.png")
    plt.close(fig)

    hours = pd.to_datetime(test["tgt_time"]).hour
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.27
    series = {"persistence": base["persistence"], **ens}
    for i, (name, pr) in enumerate(series.items()):
        rm = [np.sqrt(np.mean((pr[hours == h] - y[hours == h]) ** 2)) if (hours == h).any() else np.nan
              for h in range(24)]
        ax.bar(np.arange(24) + (i - 1) * w, rm, width=w, label=name, color=COLORS.get(name, "#999999"))
    ax.set_xlabel("giờ của bước target")
    ax.set_ylabel("RMSE (Wh)")
    ax.set_title("Sai số theo giờ trong ngày (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig / "error_by_hour.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in ("transformer", "lstm"):
        for s in seeds:
            hp = p(cfg, "output_dir", "models", name, f"seed{s}", "history.csv")
            if hp.exists():
                h = pd.read_csv(hp)
                ax[0].plot(h["epoch"], h["train_loss"], color=COLORS[name], alpha=0.6)
                ax[1].plot(h["epoch"], h["val_rmse_wh"], color=COLORS[name], alpha=0.6,
                           label=name if s == seeds[0] else None)
    ax[0].set_title("Train loss (đã scale)")
    ax[0].set_yscale("log")
    ax[1].set_title("Val RMSE (Wh)")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_fig / "training_curves.png")
    plt.close(fig)

    # ------------------------------------------------------------------ 6. báo cáo markdown
    show = summary[["model", "kind", "n_seeds", "n_params", "rmse_mean", "rmse_std", "mae_mean", "mape_mean",
                    "r2_mean"]].copy()
    lines = ["# Kết quả đánh giá (tự sinh bởi `python -m src.evaluate`)", "",
             f"Test: {len(y)} mẫu, dự đoán Appliances tại t+{cfg['window']['horizon']}, đơn vị Wh.", "",
             "| model | loại | #seed | #params | RMSE | ±std | MAE | MAPE % | R² |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in show.iterrows():
        lines.append(f"| {r.model} | {r.kind} | {int(r.n_seeds)} | {int(r.n_params)} | {r.rmse_mean:.2f} | "
                     f"{r.rmse_std:.2f} | {r.mae_mean:.2f} | {r.mape_mean:.1f} | {r.r2_mean:.3f} |")
    lines += ["", "## So sánh (block bootstrap, ΔRMSE = a − b, <0 nghĩa a tốt hơn)", "",
              "Lưu ý: dòng `*_ensemble_vs_*` so ensemble 3 seed với MỘT baseline nên có lợi cho ensemble; "
              "dòng `*_seedS_vs_ridge` so từng mô hình đơn lẻ.", "",
              "| cặp | ΔRMSE | CI95 | khác biệt có ý nghĩa? |", "|---|---|---|---|"]
    for k_, v in comps.items():
        lines.append(f"| {k_} | {v['delta_rmse']:.2f} | [{v['ci95_low']:.2f}, {v['ci95_high']:.2f}] | "
                     f"{'có' if v['significant'] else 'không'} |")
    lines += ["", "## Tiêu chí thành công", ""]
    lines += [f"- {'PASS' if c['pass'] else 'FAIL'} — {c['id']}: {c['detail']}" for c in checks]
    (p(cfg, "output_dir") / "results_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
