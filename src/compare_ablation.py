"""So sánh bản ĐẦY ĐỦ feature với bản bỏ nhóm nhiệt độ (ablation D21).

Chạy (sau khi đã chạy cả hai pipeline và `evaluate` cho cả hai):
    python -m src.compare_ablation --abl-out outputs_no_temp
Đọc:  <outputs>/predictions/*, <outputs>/metrics/per_seed_metrics.csv, <outputs>/baselines/test_predictions.npz
      (của bản đầy đủ = paths.output_dir trong config; và của bản ablation = --abl-out)
Ghi:  <abl-out>/ablation_comparison.md, ablation_comparison.csv, ablation_comparison.png

Quy tắc đọc: lựa chọn cấu hình (có dùng nhiệt độ hay không) phải dựa trên VAL. Test chỉ để báo cáo.
Δ = RMSE(bỏ nhiệt độ) - RMSE(đầy đủ). Δ < 0 nghĩa là bỏ nhiệt độ cho kết quả tốt hơn.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import add_common_args, config_from_args, p
from .metrics import block_bootstrap_delta_rmse, regression_metrics

MODELS = ("transformer", "lstm")


def _load_preds(out_dir, model, seeds):
    preds, y = {}, None
    for s in seeds:
        f = Path(out_dir) / "predictions" / f"{model}_seed{s}_test.npz"
        if not f.exists():
            continue
        z = np.load(f)
        preds[s] = z["pred"].astype(np.float64)
        y = z["y_raw"].astype(np.float64) if y is None else y
    return preds, y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--abl-out", required=True, help="thư mục outputs của bản ablation, vd. outputs_no_temp")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    full_out = p(cfg, "output_dir")
    abl_out = Path(args.abl_out)
    if not abl_out.is_absolute():
        abl_out = full_out.parent / abl_out
    seeds, ec = cfg["train"]["seeds"], cfg["eval"]

    rows, boots = [], {}
    series = {}  # cho hình

    for model in MODELS:
        pf, yf = _load_preds(full_out, model, seeds)
        pa, ya = _load_preds(abl_out, model, seeds)
        common = sorted(set(pf) & set(pa))
        if not common:
            print(f"[WARN] {model}: không có seed chung giữa hai bản, bỏ qua")
            continue
        if not np.allclose(yf, ya):
            raise SystemExit("y_raw của hai bản khác nhau: hai bản không dùng cùng tập test, không so sánh được.")
        y = yf
        res = {}
        for tag, preds in (("đầy_đủ", pf), ("bỏ_nhiệt_độ", pa)):
            per = [regression_metrics(y, preds[s]) for s in common]
            ens = regression_metrics(y, np.mean([preds[s] for s in common], axis=0))
            res[tag] = {"per": per, "ens": ens, "ens_pred": np.mean([preds[s] for s in common], axis=0)}
            # val từ per_seed_metrics.csv
            ps_path = (full_out if tag == "đầy_đủ" else abl_out) / "metrics" / "per_seed_metrics.csv"
            val_rmse = np.nan
            if ps_path.exists():
                d = pd.read_csv(ps_path)
                d = d[(d.model == model) & (d.split == "val") & (d.seed.isin(common))]
                if len(d):
                    val_rmse = float(d["rmse"].mean())
            rows.append({
                "model": model, "biến_thể": tag, "n_seed": len(common),
                "val_rmse_TB": val_rmse,
                "test_rmse_TB": float(np.mean([m["rmse"] for m in per])),
                "test_rmse_std": float(np.std([m["rmse"] for m in per])),
                "test_mae_TB": float(np.mean([m["mae"] for m in per])),
                "test_r2_TB": float(np.mean([m["r2"] for m in per])),
                "test_rmse_ensemble": ens["rmse"]})
        boots[f"{model}_ensemble"] = block_bootstrap_delta_rmse(
            y, res["bỏ_nhiệt_độ"]["ens_pred"], res["đầy_đủ"]["ens_pred"], ec["bootstrap_block"], ec["bootstrap_n"])
        for s in common:
            boots[f"{model}_seed{s}"] = block_bootstrap_delta_rmse(
                y, pa[s], pf[s], ec["bootstrap_block"], ec["bootstrap_n"])
        series[model] = res

    # Ridge (baseline tuyến tính, nhạy với drift nhất)
    bf = full_out / "baselines" / "test_predictions.npz"
    ba = abl_out / "baselines" / "test_predictions.npz"
    if bf.exists() and ba.exists():
        zf, za = np.load(bf), np.load(ba)
        y = zf["y_raw"].astype(np.float64)
        for tag, z in (("đầy_đủ", zf), ("bỏ_nhiệt_độ", za)):
            m = regression_metrics(y, z["ridge"])
            vr = np.nan
            mp = (full_out if tag == "đầy_đủ" else abl_out) / "baselines" / "metrics.json"
            if mp.exists():
                import json
                vr = json.load(open(mp, encoding="utf-8"))["val"]["ridge"]["rmse"]
            rows.append({"model": "ridge", "biến_thể": tag, "n_seed": 1, "val_rmse_TB": vr,
                         "test_rmse_TB": m["rmse"], "test_rmse_std": 0.0, "test_mae_TB": m["mae"],
                         "test_r2_TB": m["r2"], "test_rmse_ensemble": m["rmse"]})
        boots["ridge"] = block_bootstrap_delta_rmse(y, za["ridge"], zf["ridge"], ec["bootstrap_block"], ec["bootstrap_n"])

    if not rows:
        raise SystemExit("Không đọc được kết quả nào. Đã chạy evaluate cho cả hai bản chưa?")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(abl_out / "ablation_comparison.csv", index=False)

    # ---------------------------------------------------------------- hình
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(tbl["model"].unique())
    x = np.arange(len(names))
    for ax, col, title in ((axes[0], "val_rmse_TB", "Val RMSE (dùng để CHỌN)"),
                           (axes[1], "test_rmse_TB", "Test RMSE (chỉ để BÁO CÁO)")):
        for k, (tag, color) in enumerate((("đầy_đủ", "#4c78a8"), ("bỏ_nhiệt_độ", "#f58518"))):
            vals = [float(tbl[(tbl.model == m) & (tbl.biến_thể == tag)][col].iloc[0]) for m in names]
            errs = [float(tbl[(tbl.model == m) & (tbl.biến_thể == tag)]["test_rmse_std"].iloc[0]) if col == "test_rmse_TB" else 0
                    for m in names]
            ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, yerr=errs, color=color, label=tag.replace("_", " "), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.set_ylabel("RMSE (Wh)")
        lo = np.nanmin(tbl[col]) - 6
        ax.set_ylim(max(lo, 0), np.nanmax(tbl[col]) + 4)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Ablation D21: đầy đủ feature và bỏ nhóm nhiệt độ (trục y không bắt đầu từ 0)")
    fig.tight_layout()
    fig.savefig(abl_out / "ablation_comparison.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- báo cáo markdown
    L = ["# Ablation D21: bỏ nhóm nhiệt độ (tự sinh bởi `python -m src.compare_ablation`)", "",
         "Nhóm bị bỏ: `T1`–`T9`, `T_out`, `Tdewpoint`. Cùng tập test, cùng cửa sổ, cùng seed. "
         "Δ = RMSE(bỏ nhiệt độ) − RMSE(đầy đủ); Δ < 0 nghĩa là **bỏ nhiệt độ tốt hơn**.", "",
         "| model | biến thể | #seed | val RMSE | test RMSE (TB ± std) | test MAE | test R² | test RMSE ensemble |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in tbl.iterrows():
        L.append(f"| {r.model} | {r.biến_thể.replace('_', ' ')} | {int(r.n_seed)} | {r.val_rmse_TB:.2f} | "
                 f"{r.test_rmse_TB:.2f} ± {r.test_rmse_std:.2f} | {r.test_mae_TB:.2f} | {r.test_r2_TB:.3f} | "
                 f"{r.test_rmse_ensemble:.2f} |")
    L += ["", "## Kiểm định (block bootstrap trên test)", "",
          "| cặp | ΔRMSE | CI95 | có ý nghĩa? | hướng |", "|---|---|---|---|---|"]
    for k, v in boots.items():
        hd = "bỏ nhiệt độ tốt hơn" if v["delta_rmse"] < 0 else "đầy đủ tốt hơn"
        L.append(f"| {k} | {v['delta_rmse']:+.2f} | [{v['ci95_low']:+.2f}, {v['ci95_high']:+.2f}] | "
                 f"{'có' if v['significant'] else 'không'} | {hd if v['significant'] else 'chưa có bằng chứng khác biệt'} |")

    # kết luận máy đọc được, thận trọng
    L += ["", "## Kết luận tự động (đọc cùng các bảng trên, đừng đọc riêng lẻ)", ""]
    for m in names:
        a = tbl[(tbl.model == m) & (tbl.biến_thể == "bỏ_nhiệt_độ")].iloc[0]
        f = tbl[(tbl.model == m) & (tbl.biến_thể == "đầy_đủ")].iloc[0]
        key = "ridge" if m == "ridge" else f"{m}_ensemble"
        b = boots.get(key)
        dv, dt = a.val_rmse_TB - f.val_rmse_TB, a.test_rmse_TB - f.test_rmse_TB
        sig = "có ý nghĩa thống kê" if b and b["significant"] else "KHÔNG có ý nghĩa thống kê"
        L.append(f"- **{m}**: Δval = {dv:+.2f} Wh, Δtest = {dt:+.2f} Wh ({sig}).")
        if np.isfinite(dv) and np.sign(dv) != np.sign(dt) and abs(dt) > 0.5:
            L.append(f"  - Val và test cho hướng ngược nhau với {m}: dấu hiệu drift. Chọn cấu hình theo **val**, "
                     "nêu test như một quan sát.")
    L += ["", "## Giới hạn khi diễn giải",
          "- Chỉ một giai đoạn test và 3 seed; chênh lệch nhỏ hơn dao động giữa seed thì không nên diễn giải.",
          "- Bỏ nhiệt độ cũng bỏ luôn tín hiệu mùa (nhiệt độ ngoài trời tương quan với thời điểm trong năm), "
          "nên kết quả không tách riêng được “ngoại suy do drift” khỏi “mất thông tin có ích”.",
          "- Không dùng test để quyết định cấu hình chính. Cấu hình chính chọn theo val (cột `val RMSE`)."]
    (abl_out / "ablation_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {abl_out / 'ablation_comparison.md'}")


if __name__ == "__main__":
    main()
