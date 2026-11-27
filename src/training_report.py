"""Báo cáo ĐẦY ĐỦ quá trình huấn luyện của mọi (model, seed): đường cong, epoch tốt nhất, lý do dừng.

Chạy:  python -m src.training_report
Sinh ra trong outputs/figures/ và outputs/models/:
    training_overview.png      - tất cả run trên cùng một hình: train loss, val RMSE, learning rate
    training_per_run.png       - lưới từng run (hàng = model, cột = seed), có đánh dấu epoch tốt nhất và điểm dừng
    training_runs_table.csv    - bảng tóm tắt + lý do dừng từng run
Chỉ cần pandas / matplotlib / numpy (không cần torch), nên dùng được trong notebook.
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

from .config import add_common_args, config_from_args, p
from .utils import ensure_dir, load_json

COLORS = {"transformer": "#d62728", "lstm": "#1f77b4"}
MIN_DELTA = 1e-6  # phải trùng với điều kiện "improved" trong train.py


def replay_run(df, patience, max_epochs):
    """Tái hiện đúng logic early stopping của train.py trên history để tìm epoch tốt nhất và lý do dừng."""
    best, best_ep, bad = float("inf"), None, 0
    stop_ep, reason = int(df["epoch"].iloc[-1]), None
    for _, r in df.iterrows():
        v, ep = float(r["val_rmse_wh"]), int(r["epoch"])
        if v < best - MIN_DELTA:
            best, best_ep, bad = v, ep, 0
        else:
            bad += 1
        if bad >= patience:
            stop_ep = ep
            reason = (f"Early stopping: val RMSE không cải thiện {patience} epoch liên tiếp "
                      f"(tốt nhất ở epoch {best_ep}, dừng ở epoch {ep})")
            break
    n = len(df)
    if reason is None:
        if n >= max_epochs:
            reason = f"Hết số epoch tối đa ({max_epochs}); val RMSE vẫn có thể còn giảm, cân nhắc tăng epochs"
        else:
            reason = f"Chưa kết thúc / bị ngắt ở epoch {n} (chưa tới early stopping hay epoch tối đa)"
    return best_ep, best, stop_ep, reason


def load_runs(cfg):
    runs = []
    root = p(cfg, "output_dir", "models")
    for hp in sorted(root.glob("*/seed*/history.csv")):
        model, seed = hp.parent.parent.name, hp.parent.name
        df = pd.read_csv(hp)
        if df.empty:
            continue
        snap = hp.parent / "config_snapshot.json"
        tcfg = load_json(snap)["train"] if snap.exists() else cfg["train"]
        summ = hp.parent / "train_summary.json"
        info = load_json(summ) if summ.exists() else {}
        runs.append({"model": model, "seed": seed, "df": df, "patience": tcfg["patience"],
                     "max_epochs": tcfg["epochs"], "info": info, "finished": summ.exists()})
    return runs


def runs_table(runs):
    rows = []
    for r in runs:
        df = r["df"]
        best_ep, best, stop_ep, reason = replay_run(df, r["patience"], r["max_epochs"])
        lr = df["lr"].to_numpy()
        n_lr_drops = int((lr[1:] < lr[:-1] - 1e-15).sum())
        first_val = float(df["val_rmse_wh"].iloc[0])
        tl = df["train_loss"].to_numpy()
        row = {
            "model": r["model"], "seed": r["seed"], "epochs_chạy": int(len(df)), "epoch_tốt_nhất": best_ep,
            "val_RMSE_epoch1": round(first_val, 2), "val_RMSE_tốt_nhất": round(best, 2),
            "cải_thiện_so_với_epoch1": round(first_val - best, 2),
            "train_loss_đầu": round(float(tl[0]), 4), "train_loss_cuối": round(float(tl[-1]), 4),
            "lần_giảm_lr": n_lr_drops, "lr_cuối": float(lr[-1]),
            "thời_gian_s": float(df["elapsed_s"].iloc[-1]) if "elapsed_s" in df else r["info"].get("train_seconds"),
            "n_params": r["info"].get("n_params"), "lý_do_dừng": reason,
        }
        # dấu hiệu overfit: train loss tiếp tục giảm sau epoch tốt nhất trong khi val RMSE tăng
        # So trung bình val RMSE SAU epoch tốt nhất với trung bình các epoch ngay TRƯỚC đó (cùng độ dài).
        # (So với chính giá trị tốt nhất sẽ luôn thấy "tăng", vì đó là điểm thấp nhất của một chuỗi nhiễu.)
        after = df[df["epoch"] > best_ep]["val_rmse_wh"]
        before = df[(df["epoch"] <= best_ep) & (df["epoch"] > best_ep - max(len(after), 3))]["val_rmse_wh"]
        train_still_down = bool(df["train_loss"].iloc[-1] < tl[best_ep - 1])
        if len(after) >= 3 and after.mean() > before.mean() + 1.0 and train_still_down:
            row["nhận_xét"] = "Val RMSE có xu hướng TĂNG sau epoch tốt nhất trong khi train loss vẫn giảm: nghi overfit"
        elif len(after) >= 3:
            row["nhận_xét"] = ("Val RMSE đi ngang, chỉ dao động quanh mức tốt nhất (đã bão hoà); "
                               "epoch tốt nhất một phần là điểm thấp nhất của dao động ngẫu nhiên, không hẳn một đỉnh rõ")
        else:
            row["nhận_xét"] = "Dừng gần epoch tốt nhất"
        rows.append(row)
    return pd.DataFrame(rows)


def reference_lines(cfg):
    mp = p(cfg, "output_dir", "baselines", "metrics.json")
    if not mp.exists():
        return {}
    val = load_json(mp).get("val", {})
    return {k: val[k]["rmse"] for k in ("ridge", "moving_avg_6") if k in val}


def plot_overview(runs, refs, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    seeds = sorted({r["seed"] for r in runs})
    lss = {s: ["-", "--", ":", "-."][i % 4] for i, s in enumerate(seeds)}
    for r in runs:
        df, c, ls = r["df"], COLORS.get(r["model"], "#555"), lss[r["seed"]]
        lab = f"{r['model']} {r['seed']}"
        best_ep, best, stop_ep, _ = replay_run(df, r["patience"], r["max_epochs"])
        ax[0].plot(df["epoch"], df["train_loss"], color=c, ls=ls, alpha=0.85, label=lab)
        ax[1].plot(df["epoch"], df["val_rmse_wh"], color=c, ls=ls, alpha=0.85, label=lab)
        ax[2].plot(df["epoch"], df["lr"], color=c, ls=ls, alpha=0.85, label=lab)
        ax[1].scatter([best_ep], [best], color=c, s=40, zorder=4, edgecolor="black")
        ax[1].scatter([stop_ep], [float(df.loc[df["epoch"] == stop_ep, "val_rmse_wh"].iloc[0])], color=c,
                      marker="x", s=55, zorder=4)
    for name, v in refs.items():
        ax[1].axhline(v, color="grey", ls=":", lw=1.2)
        ax[1].text(ax[1].get_xlim()[1], v, f" {name} (val)", fontsize=8, va="bottom", ha="right", color="grey")
    ax[0].set_title("Train loss (đã scale, trục log)")
    ax[0].set_yscale("log")
    ax[1].set_title("Val RMSE (Wh)   ● = epoch tốt nhất   ✕ = điểm dừng")
    ax[2].set_title("Learning rate")
    ax[2].set_yscale("log")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.25)
    ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Toàn bộ quá trình huấn luyện: mọi model và mọi seed", fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_per_run(runs, refs, path=None):
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    fig, axes = plt.subplots(len(models), len(seeds), figsize=(5.2 * len(seeds), 3.6 * len(models)), squeeze=False)
    by = {(r["model"], r["seed"]): r for r in runs}
    for i, m in enumerate(models):
        for j, s in enumerate(seeds):
            ax = axes[i, j]
            r = by.get((m, s))
            if r is None:
                ax.axis("off")
                continue
            df, c = r["df"], COLORS.get(m, "#555")
            best_ep, best, stop_ep, _ = replay_run(df, r["patience"], r["max_epochs"])
            ax.plot(df["epoch"], df["train_loss"], color="#2ca02c", label="train loss")
            ax.set_yscale("log")
            ax.set_ylabel("train loss", color="#2ca02c", fontsize=8)
            ax2 = ax.twinx()
            ax2.plot(df["epoch"], df["val_rmse_wh"], color=c, label="val RMSE")
            ax2.axvline(best_ep, color="black", ls="--", lw=0.9)
            ax2.scatter([best_ep], [best], color=c, edgecolor="black", s=45, zorder=4)
            ax2.axvspan(best_ep, stop_ep, color="orange", alpha=0.12)
            for name, v in refs.items():
                ax2.axhline(v, color="grey", ls=":", lw=1)
            ax2.set_ylabel("val RMSE (Wh)", color=c, fontsize=8)
            ax.set_title(f"{m} {s}: tốt nhất ep {best_ep} ({best:.1f} Wh), dừng ep {stop_ep}", fontsize=9)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.2)
    fig.suptitle("Từng run: đường đứt = epoch tốt nhất; vùng cam = các epoch chờ cải thiện (patience); "
                 "đường chấm xám = Ridge / moving_avg_6 trên val", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    cfg = config_from_args(ap.parse_args(argv))
    matplotlib.use("Agg")
    runs = load_runs(cfg)
    if not runs:
        raise SystemExit("Không thấy history.csv nào. Chạy `python -m src.train` trước.")
    refs = reference_lines(cfg)
    fig_dir = ensure_dir(p(cfg, "output_dir", "figures"))
    plot_overview(runs, refs, fig_dir / "training_overview.png")
    plot_per_run(runs, refs, fig_dir / "training_per_run.png")
    tbl = runs_table(runs)
    out_csv = ensure_dir(p(cfg, "output_dir", "models")) / "training_runs_table.csv"
    tbl.to_csv(out_csv, index=False)
    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        print(tbl[["model", "seed", "epochs_chạy", "epoch_tốt_nhất", "val_RMSE_tốt_nhất", "lý_do_dừng"]].to_string(index=False))
    print(f"[OK] {fig_dir / 'training_overview.png'}\n[OK] {fig_dir / 'training_per_run.png'}\n[OK] {out_csv}")


if __name__ == "__main__":
    main()
