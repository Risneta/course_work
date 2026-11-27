"""EDA tự động: kiểm tra chất lượng dữ liệu, phân phối, chu kỳ, tương quan, drift giữa các split.

Chạy:  python -m src.eda
Sinh ra outputs/eda/*.png, *.csv, eda_report.json và eda_summary.md (bản tóm tắt để đọc & quyết định).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import add_common_args, config_from_args, p
from .data import load_raw, split_bounds, validate_raw
from .metrics import regression_metrics
from .utils import ensure_dir, save_json

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    return np.array([1.0] + [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, max_lag + 1)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    cfg = config_from_args(ap.parse_args(argv))
    out = ensure_dir(p(cfg, "output_dir", "eda"))
    d = cfg["data"]
    tgt, dcol, H = d["target_col"], d["date_col"], cfg["window"]["horizon"]

    df = load_raw(p(cfg, "raw_csv"), dcol)
    val_report = validate_raw(df, cfg)
    num = df.drop(columns=[dcol])
    y = df[tgt].to_numpy(float)
    t = df[dcol]
    report = {"validation": val_report}

    # ---------------------------------------------------------------- 1. thống kê mô tả
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["kurtosis"] = num.kurt()
    desc["zero_pct"] = (num == 0).mean() * 100
    desc.to_csv(out / "descriptive_stats.csv")

    # ---------------------------------------------------------------- 2. target
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    n_out = int(((y > q3 + 1.5 * iqr) | (y < q1 - 1.5 * iqr)).sum())
    report["target"] = {
        "mean": float(y.mean()), "std": float(y.std()), "median": float(np.median(y)),
        "min": float(y.min()), "max": float(y.max()), "skew": float(stats.skew(y)),
        "kurtosis": float(stats.kurtosis(y)), "p90": float(np.percentile(y, 90)), "p99": float(np.percentile(y, 99)),
        "iqr_outlier_count": n_out, "iqr_outlier_pct": float(n_out / len(y) * 100),
        "skew_log1p": float(stats.skew(np.log1p(y))),
    }
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].hist(y, bins=80, color="#1f77b4")
    ax[0].set_title("Appliances (Wh)")
    ax[1].hist(np.log1p(y), bins=80, color="#2ca02c")
    ax[1].set_title("log1p(Appliances)")
    ax[2].boxplot(y)
    ax[2].set_title("Boxplot")
    fig.tight_layout()
    fig.savefig(out / "target_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(13, 6))
    ax[0].plot(t, y, lw=0.3)
    ax[0].plot(t, pd.Series(y).rolling(144, min_periods=1).mean(), color="red", lw=1, label="rolling mean 1 ngày")
    ax[0].set_title("Appliances toàn bộ giai đoạn")
    ax[0].legend()
    k = min(7 * 144, len(y))
    ax[1].plot(t[:k], y[:k], lw=0.7)
    ax[1].set_title("Zoom 7 ngày đầu")
    fig.tight_layout()
    fig.savefig(out / "target_timeseries.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 3. chu kỳ ngày / tuần
    tmp = pd.DataFrame({"y": y, "hour": t.dt.hour, "dow": t.dt.dayofweek})
    by_h = tmp.groupby("hour")["y"].agg(["mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    by_h.columns = ["mean", "q25", "q75"]
    by_d = tmp.groupby("dow")["y"].mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(by_h.index, by_h["mean"], marker="o")
    ax[0].fill_between(by_h.index, by_h["q25"], by_h["q75"], alpha=0.25)
    ax[0].set_title("Trung bình theo giờ (dải = IQR)")
    ax[0].set_xlabel("giờ")
    ax[1].bar(["T2", "T3", "T4", "T5", "T6", "T7", "CN"], by_d.values, color="#1f77b4")
    ax[1].set_title("Trung bình theo thứ")
    fig.tight_layout()
    fig.savefig(out / "daily_weekly_profile.png")
    plt.close(fig)
    report["seasonality"] = {
        "peak_hour": int(by_h["mean"].idxmax()), "trough_hour": int(by_h["mean"].idxmin()),
        "hour_peak_over_trough": float(by_h["mean"].max() / by_h["mean"].min()),
        "weekday_mean": float(tmp[tmp.dow < 5]["y"].mean()), "weekend_mean": float(tmp[tmp.dow >= 5]["y"].mean()),
    }

    # ---------------------------------------------------------------- 4. ACF
    max_lag = min(288, len(y) // 4)
    a = acf(y, max_lag)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(range(len(a)), a, width=1.0)
    ax.axvline(H, color="red", ls="--", label=f"horizon H={H}")
    ax.set_title("ACF của Appliances (1 bước = 10 phút)")
    ax.set_xlabel("lag")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "acf_target.png")
    plt.close(fig)
    report["acf"] = {f"lag_{k}": float(a[k]) for k in (1, 3, 6, 36, 144) if k <= max_lag}

    # ---------------------------------------------------------------- 5. tương quan
    corr = num.corr()
    order = corr[tgt].abs().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.loc[order, order].values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Ma trận tương quan (sắp theo |corr| với target)")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png")
    plt.close(fig)

    feat_cols = [c for c in num.columns if c != tgt]
    c0 = corr[tgt].drop(tgt)
    cH = pd.Series({c: np.corrcoef(num[c].to_numpy()[:-H], y[H:])[0, 1] for c in feat_cols})
    cdf = pd.DataFrame({"corr_same_time": c0, f"corr_feature_t_vs_target_t+{H}": cH}).sort_values(
        f"corr_feature_t_vs_target_t+{H}", key=np.abs, ascending=False)
    cdf.to_csv(out / "target_correlations.csv")
    top = cdf.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top[f"corr_feature_t_vs_target_t+{H}"], color="#1f77b4")
    ax.set_title(f"Top 15 |corr(feature[t], Appliances[t+{H}])|")
    fig.tight_layout()
    fig.savefig(out / "target_correlation_top.png")
    plt.close(fig)
    report["top_correlated_features"] = cdf.head(5)[f"corr_feature_t_vs_target_t+{H}"].round(4).to_dict()
    report["noise_vars"] = {c: float(corr.loc[c, tgt]) for c in ("rv1", "rv2") if c in corr.index}

    # ---------------------------------------------------------------- 6. phân phối từng feature
    cols = list(num.columns)
    nc = 6
    nr = int(np.ceil(len(cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(14, 2.1 * nr))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(num[c], bins=40, color="#7f7f7f")
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "feature_distributions.png")
    plt.close(fig)

    # ---------------------------------------------------------------- 7. drift giữa các split
    train_end, val_end = split_bounds(len(df), cfg)
    segs = {"train": num.iloc[:train_end], "val": num.iloc[train_end:val_end], "test": num.iloc[val_end:]}
    mu_tr, sd_tr = segs["train"].mean(), segs["train"].std().replace(0, 1)
    drift = pd.DataFrame({k: (v.mean() - mu_tr) / sd_tr for k, v in segs.items()})
    drift.to_csv(out / "split_drift_zscore.csv")
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(drift[["val", "test"]].values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(drift)))
    ax.set_yticklabels(drift.index, fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["val", "test"])
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    ax.set_title("Lệch trung bình so với train (đơn vị std train)")
    fig.tight_layout()
    fig.savefig(out / "split_drift.png")
    plt.close(fig)
    ks = stats.ks_2samp(segs["train"][tgt], segs["test"][tgt])
    report["split_target"] = {
        k: {"mean": float(v[tgt].mean()), "std": float(v[tgt].std()), "n": int(len(v))} for k, v in segs.items()}
    report["split_target"]["ks_train_vs_test"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    report["max_abs_drift_test"] = {c: float(v) for c, v in drift["test"].abs().sort_values(ascending=False).head(5).items()}

    # ---------------------------------------------------------------- 8. mức khó của bài toán: persistence
    sl = slice(val_end, None)
    pers = regression_metrics(y[sl][H:], y[sl][:-H])
    report["persistence_preview_test"] = pers

    save_json(report, out / "eda_report.json")

    # ---------------------------------------------------------------- 9. summary.md (đọc để ra quyết định)
    tr = report["target"]
    notes = []
    if abs(tr["skew"]) > 1:
        notes.append(f"Target lệch phải mạnh (skew={tr['skew']:.2f}; sau log1p còn {tr['skew_log1p']:.2f}) "
                     f"-> cân nhắc `data.target_transform: log1p` (quyết định mở D3).")
    if tr["iqr_outlier_pct"] > 5:
        notes.append(f"{tr['iqr_outlier_pct']:.1f}% điểm là outlier theo IQR; đây là các đỉnh tiêu thụ thật, "
                     f"KHÔNG nên xoá. Dùng loss Huber là một lựa chọn (quyết định mở D4).")
    if any(abs(v) < 0.05 for v in report["noise_vars"].values()):
        notes.append("rv1/rv2 gần như không tương quan với target -> giữ trong `drop_cols`.")
    zl = desc.loc["lights", "zero_pct"] if "lights" in desc.index else None
    if zl is not None and zl > 50:
        notes.append(f"`lights` bằng 0 ở {zl:.0f}% số dòng (biến rời rạc, nhiều số 0).")
    md = max(report["max_abs_drift_test"].values())
    if md > 1:
        notes.append(f"Có drift giữa train và test (lớn nhất {md:.2f} std) -> mô hình có thể giảm hiệu năng ở test; "
                     f"ghi vào báo cáo kết quả.")
    if val_report["warnings"]:
        notes += [f"[Dữ liệu] {w}" for w in val_report["warnings"]]
    lines = [
        "# EDA summary (tự sinh bởi `python -m src.eda`)", "",
        f"- Số dòng: **{val_report['n_rows']}**, giai đoạn {val_report['date_min']} -> {val_report['date_max']}",
        f"- Thiếu: {val_report['n_missing_cells']} ô | timestamp trùng: {val_report['n_duplicate_timestamps']} | "
        f"bước thời gian không đều: {val_report['n_irregular_steps']}",
        f"- Target `{tgt}`: mean={tr['mean']:.1f}, median={tr['median']:.1f}, std={tr['std']:.1f}, "
        f"min={tr['min']:.0f}, max={tr['max']:.0f}, P99={tr['p99']:.0f}, skew={tr['skew']:.2f}",
        f"- Giờ cao điểm: {report['seasonality']['peak_hour']}h; giờ thấp nhất: {report['seasonality']['trough_hour']}h "
        f"(tỉ lệ {report['seasonality']['hour_peak_over_trough']:.2f}x)",
        "- ACF: " + ", ".join(f"{k}={v:.2f}" for k, v in report["acf"].items()),
        "- Top feature theo |corr(feature[t], target[t+H])|: " + ", ".join(
            f"{k}={v:.2f}" for k, v in report["top_correlated_features"].items()),
        f"- Persistence trên đoạn test (t -> t+{H}): RMSE={pers['rmse']:.1f} Wh, R2={pers['r2']:.3f} "
        f"(mốc tối thiểu mà mô hình phải vượt)", "",
        "## Phát hiện & gợi ý quyết định", ""] + [f"{i + 1}. {n}" for i, n in enumerate(notes)] + [
        "", "Hình: `target_distribution.png`, `target_timeseries.png`, `daily_weekly_profile.png`, `acf_target.png`, "
        "`correlation_heatmap.png`, `target_correlation_top.png`, `feature_distributions.png`, `split_drift.png`."]
    (out / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] đã lưu vào {out}")


if __name__ == "__main__":
    main()
