"""Theo dõi quá trình huấn luyện (đọc history.csv do train.py ghi sau mỗi epoch).

Chạy (mở terminal thứ hai trong lúc `python -m src.train` đang chạy):
    python -m src.monitor                 # in bảng + lưu hình 1 lần
    python -m src.monitor --watch 10      # tự làm mới mỗi 10 giây (Ctrl+C để dừng)
    python -m src.monitor --live          # mở cửa sổ biểu đồ cập nhật trực tiếp
"""
import argparse
import os
import time

import pandas as pd

from .config import add_common_args, config_from_args, p
from .utils import ensure_dir, load_json

COLORS = {"transformer": "#d62728", "lstm": "#1f77b4"}


def collect(cfg):
    runs = {}
    root = p(cfg, "output_dir", "models")
    for hp in sorted(root.glob("*/seed*/history.csv")):
        try:
            df = pd.read_csv(hp)
        except Exception:  # file đang được ghi dở
            continue
        if df.empty:
            continue
        model, seed = hp.parent.parent.name, hp.parent.name
        runs[(model, seed)] = (df, (hp.parent / "train_summary.json").exists())
    return runs


def reference_lines(cfg):
    """Val RMSE của baseline (nếu đã chạy baselines) để đối chiếu trên đồ thị."""
    mp = p(cfg, "output_dir", "baselines", "metrics.json")
    if not mp.exists():
        return {}
    val = load_json(mp).get("val", {})
    return {k: val[k]["rmse"] for k in ("ridge", "moving_avg_6") if k in val}


def summarize(runs, patience):
    rows = []
    for (model, seed), (df, done) in runs.items():
        i = df["val_rmse_wh"].idxmin()
        best_ep, best = int(df.loc[i, "epoch"]), float(df.loc[i, "val_rmse_wh"])
        last = df.iloc[-1]
        since = int(last["epoch"]) - best_ep
        if done:
            status = "xong"
        else:
            status = f"đang chạy (còn {max(patience - since, 0)} epoch tới early-stop)"
        rows.append({"model": model, "seed": seed, "epoch": int(last["epoch"]), "train_loss": round(last["train_loss"], 4),
                     "val_RMSE": round(float(last["val_rmse_wh"]), 2), "best_val": round(best, 2),
                     "best_ep": best_ep, "lr": f"{last['lr']:.1e}",
                     "time_s": round(float(last.get("elapsed_s", float("nan"))), 0), "status": status})
    return pd.DataFrame(rows)


def draw(runs, refs, axes):
    for ax in axes:
        ax.clear()
        ax.grid(alpha=0.25)
    ax_l, ax_v = axes
    for (model, seed), (df, done) in runs.items():
        c = COLORS.get(model, "#555555")
        ls = "-" if done else "--"
        ax_l.plot(df["epoch"], df["train_loss"], color=c, ls=ls, alpha=0.8)
        ax_v.plot(df["epoch"], df["val_rmse_wh"], color=c, ls=ls, alpha=0.8, label=f"{model} {seed}")
        i = df["val_rmse_wh"].idxmin()
        ax_v.scatter(df.loc[i, "epoch"], df.loc[i, "val_rmse_wh"], color=c, s=25, zorder=3)
    for name, v in refs.items():
        ax_v.axhline(v, color="grey", ls=":", lw=1)
        ax_v.text(ax_v.get_xlim()[0], v, f" {name} (val)", fontsize=7, va="bottom", color="grey")
    ax_l.set_title("Train loss (đã scale)")
    ax_l.set_yscale("log")
    ax_v.set_title("Val RMSE (Wh)  —  chấm = epoch tốt nhất; nét đứt = đang chạy")
    ax_v.legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xlabel("epoch")


def render(cfg, patience, fig=None, axes=None, save=True):
    runs = collect(cfg)
    if not runs:
        print("Chưa có history.csv nào. Hãy chạy `python -m src.train` trước (hoặc đợi epoch đầu tiên xong).")
        return False
    tbl = summarize(runs, patience)
    print(tbl.to_string(index=False))
    if fig is not None:
        draw(runs, reference_lines(cfg), axes)
        fig.tight_layout()
    if save:
        out = ensure_dir(p(cfg, "output_dir", "figures")) / "training_monitor.png"
        fig.savefig(out, dpi=120)
        print(f"[hình] {out}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--watch", type=float, default=0, help="làm mới mỗi N giây (0 = chạy 1 lần)")
    ap.add_argument("--live", action="store_true", help="mở cửa sổ biểu đồ cập nhật trực tiếp")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    patience = cfg["train"]["patience"]

    import matplotlib
    if not args.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if args.live:
        plt.ion()
        interval = args.watch or 5
        try:
            while plt.fignum_exists(fig.number):
                os.system("cls" if os.name == "nt" else "clear")
                render(cfg, patience, fig, axes)
                fig.canvas.draw_idle()
                plt.pause(interval)
        except KeyboardInterrupt:
            pass
        return
    if args.watch > 0:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                render(cfg, patience, fig, axes)
                print(f"\n(làm mới mỗi {args.watch:g}s, Ctrl+C để dừng)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
        return
    render(cfg, patience, fig, axes)


if __name__ == "__main__":
    main()
