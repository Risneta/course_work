"""Phân tích attention map của Transformer + permutation importance (so sánh với LSTM).

Chạy:  python -m src.attention --seed 0
Lưu ý: attention KHÔNG tự động là giải thích nhân quả; permutation importance được thêm vào để đối chiếu.
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
from .models import load_checkpoint
from .train import get_device, load_split, predict
from .utils import ensure_dir, load_json, save_json

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def collect_attention(model, X, device, batch_size=256):
    """Trả về dict thống kê attention tổng hợp trên toàn bộ X."""
    model.eval()
    n_layers = len(model.layers)
    sums = None
    cls_per_sample = []
    ent_sum, ent_n = None, 0
    roll_sum = None
    max_rowsum_err = 0.0
    first = 1 if model.has_cls else 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        _, attn = model.forward_with_attention(xb)  # list (B, heads, S, S)
        A = torch.stack(attn, dim=1)  # (B, layers, heads, S, S)
        max_rowsum_err = max(max_rowsum_err, float((A.sum(-1) - 1).abs().max()))
        S = A.size(-1)
        # tổng attention matrix
        s = A.sum(0)
        sums = s if sums is None else sums + s
        # attention của query "đại diện" tới các bước thời gian: CLS (hàng 0) hoặc trung bình mọi query
        q = A[:, :, :, 0, first:] if model.has_cls else A.mean(dim=3)
        cls_per_sample.append(q.cpu().numpy())  # (B, layers, heads, L)
        # entropy mỗi head (trung bình trên các query)
        ent = -(A.clamp_min(1e-12).log() * A).sum(-1).mean(-1)  # (B, layers, heads)
        ent_sum = ent.sum(0) if ent_sum is None else ent_sum + ent.sum(0)
        ent_n += A.size(0)
        # attention rollout: trung bình head, cộng identity, chuẩn hoá hàng, nhân qua các layer
        R = torch.eye(S, device=A.device).expand(A.size(0), S, S)
        for l in range(n_layers):
            a = A[:, l].mean(dim=1) + torch.eye(S, device=A.device)
            a = a / a.sum(-1, keepdim=True)
            R = torch.bmm(a, R)
        roll = R[:, 0, first:] if model.has_cls else R.mean(dim=1)
        roll_sum = roll.sum(0) if roll_sum is None else roll_sum + roll.sum(0)
    n = len(X)
    return {
        "mean_matrix": (sums / n).cpu().numpy(),                 # (layers, heads, S, S)
        "per_sample_q": np.concatenate(cls_per_sample, axis=0),  # (N, layers, heads, L)
        "entropy": (ent_sum / ent_n).cpu().numpy(),              # (layers, heads)
        "rollout": (roll_sum / n).cpu().numpy(),                 # (L,)
        "max_rowsum_err": max_rowsum_err,
        "S": int(sums.size(-1)),
    }


def permutation_importance(model, X, y_raw, sc, device, repeats, seed=0):
    rng = np.random.default_rng(seed)
    base = np.sqrt(np.mean((inverse_target(predict(model, X, device), sc) - y_raw) ** 2))
    imp = np.zeros((X.shape[2], repeats))
    for f in range(X.shape[2]):
        for r in range(repeats):
            Xp = X.copy()
            Xp[:, :, f] = X[rng.permutation(len(X))][:, :, f]
            rm = np.sqrt(np.mean((inverse_target(predict(model, Xp, device), sc) - y_raw) ** 2))
            imp[f, r] = rm - base
    return base, imp.mean(1), imp.std(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    ac, L = cfg["attention"], cfg["window"]["lookback"]
    device = get_device(cfg)
    sc = load_json(p(cfg, "processed_dir", "scaler.json"))
    feat = load_json(p(cfg, "processed_dir", "feature_names.json"))
    test = load_split(cfg, "test")
    out = ensure_dir(p(cfg, "output_dir", "attention"))

    model, _ = load_checkpoint(p(cfg, "output_dir", "models", "transformer", f"seed{args.seed}", "best.pt"), device)
    res = collect_attention(model, test["X"], device)
    n_layers, n_heads = res["entropy"].shape
    S, first = res["S"], (1 if model.has_cls else 0)
    lags_min = -(np.arange(L)[::-1]) * cfg["data"]["freq_minutes"]  # vị trí j=0 -> xa nhất

    # ------------------------------------------------------------ 1. CLS->time heatmap theo (layer, head)
    q_mean = res["per_sample_q"].mean(0)  # (layers, heads, L)
    fig, axes = plt.subplots(1, n_layers, figsize=(5.2 * n_layers, 3.4), squeeze=False)
    for l, ax in enumerate(axes[0]):
        im = ax.imshow(q_mean[l], aspect="auto", cmap="viridis")
        ax.set_title(f"Layer {l}: attention của {'CLS' if first else 'mọi query'} -> thời gian")
        ax.set_xlabel("vị trí trong window (phải = gần t nhất)")
        ax.set_ylabel("head")
        ax.set_xticks([0, L // 2, L - 1])
        ax.set_xticklabels([f"{lags_min[0]}m", f"{lags_min[L // 2]}m", "0m"])
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "attn_heads_over_time.png")
    plt.close(fig)

    # ------------------------------------------------------------ 2. trung bình theo lag + rollout
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for l in range(n_layers):
        ax.plot(lags_min, q_mean[l].mean(0), marker=".", label=f"layer {l} (TB các head)")
    ax.plot(lags_min, res["rollout"] / res["rollout"].sum(), "k--", label="rollout (chuẩn hoá)")
    ax.axhline(1 / L, color="grey", ls=":", label="đều (1/L)")
    ax.set_xlabel("độ trễ so với bước cuối của window (phút)")
    ax.set_ylabel("trọng số attention")
    ax.set_title("Attention theo độ trễ (trung bình trên test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "attn_by_lag.png")
    plt.close(fig)

    # ------------------------------------------------------------ 3. ma trận attention đầy đủ mỗi layer
    fig, axes = plt.subplots(1, n_layers, figsize=(4.8 * n_layers, 4.2), squeeze=False)
    for l, ax in enumerate(axes[0]):
        im = ax.imshow(res["mean_matrix"][l].mean(0), cmap="magma")
        ax.set_title(f"Layer {l}: attention TB (hàng=query, cột=key)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "attn_full_matrix.png")
    plt.close(fig)

    # ------------------------------------------------------------ 4. entropy mỗi head
    fig, ax = plt.subplots(figsize=(6, 3.4))
    im = ax.imshow(res["entropy"] / np.log(S), cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Entropy chuẩn hoá của từng head (1 = đều hoàn toàn)")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    for l in range(n_layers):
        for h in range(n_heads):
            ax.text(h, l, f"{res['entropy'][l, h] / np.log(S):.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "attn_entropy.png")
    plt.close(fig)

    # ------------------------------------------------------------ 5. nhóm tiêu thụ cao vs bình thường
    y = test["y_raw"]
    hi = y >= np.quantile(y, ac["high_quantile"])
    lo = y < np.median(y)
    avg_layers = res["per_sample_q"].mean(axis=(1, 2))  # (N, L)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(lags_min, avg_layers[hi].mean(0), label=f"tiêu thụ cao (>=P{int(ac['high_quantile'] * 100)})", color="#d62728")
    ax.plot(lags_min, avg_layers[lo].mean(0), label="tiêu thụ thấp (< median)", color="#1f77b4")
    ax.set_xlabel("độ trễ (phút)")
    ax.set_ylabel("attention TB (layer, head)")
    ax.set_title("Attention theo nhóm mẫu")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "attn_high_vs_low.png")
    plt.close(fig)

    # ------------------------------------------------------------ 6. vài ví dụ riêng lẻ
    tgt_col = feat.index(cfg["data"]["target_col"]) if cfg["data"]["target_col"] in feat else None
    idx = np.argsort(y)[::-1][: ac["n_examples"] // 2].tolist() + \
        np.random.default_rng(0).choice(len(y), ac["n_examples"] - ac["n_examples"] // 2, replace=False).tolist()
    fig, axes = plt.subplots(len(idx), 1, figsize=(9, 2.4 * len(idx)), sharex=True, squeeze=False)
    for ax, i in zip(axes[:, 0], idx):
        ax.bar(lags_min, avg_layers[i], width=8, color="#d62728", alpha=0.7)
        ax.set_ylabel("attention")
        if tgt_col is not None:
            ax2 = ax.twinx()
            ax2.plot(lags_min, test["hist_raw"][i], color="black", lw=1)
            ax2.set_ylabel("Appliances (Wh)")
        ax.set_title(f"mẫu test #{i}: target thật = {y[i]:.0f} Wh", fontsize=9)
    axes[-1, 0].set_xlabel("độ trễ (phút)")
    fig.tight_layout()
    fig.savefig(out / "attn_examples.png")
    plt.close(fig)

    # ------------------------------------------------------------ 7. permutation importance: Transformer vs LSTM
    perm_info = None
    if not args.skip_permutation:
        records = {}
        for name in ("transformer", "lstm"):
            ck = p(cfg, "output_dir", "models", name, f"seed{args.seed}", "best.pt")
            if not ck.exists():
                continue
            m, _ = load_checkpoint(ck, device)
            base, mu, sd = permutation_importance(m, test["X"], y, sc, device, ac["permutation_repeats"])
            records[name] = (base, mu, sd)
        if records:
            df = pd.DataFrame({"feature": feat})
            for name, (base, mu, sd) in records.items():
                df[f"{name}_delta_rmse"], df[f"{name}_std"] = mu, sd
            df.to_csv(out / "permutation_importance.csv", index=False)
            key = "transformer_delta_rmse" if "transformer_delta_rmse" in df else df.columns[1]
            top = df.sort_values(key, ascending=False).head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 5.5))
            yy = np.arange(len(top))
            if "transformer_delta_rmse" in top:
                ax.barh(yy + 0.2, top["transformer_delta_rmse"], height=0.4, color="#d62728", label="transformer")
            if "lstm_delta_rmse" in top:
                ax.barh(yy - 0.2, top["lstm_delta_rmse"], height=0.4, color="#1f77b4", label="lstm")
            ax.set_yticks(yy)
            ax.set_yticklabels(top["feature"])
            ax.set_xlabel("tăng RMSE (Wh) khi xáo trộn feature")
            ax.set_title("Permutation importance (top 15 theo Transformer)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "permutation_importance.png")
            plt.close(fig)
            perm_info = {n: {"base_rmse": float(b)} for n, (b, _, _) in records.items()}

    # ------------------------------------------------------------ 8. tóm tắt + kiểm tra
    ent_ratio = res["entropy"] / np.log(S)
    top_lags = np.argsort(res["rollout"])[::-1][:3]
    summary = {
        "seed": args.seed, "n_layers": n_layers, "n_heads": n_heads, "seq_len_with_cls": S,
        "max_rowsum_error": res["max_rowsum_err"],
        "check_rows_sum_to_1": bool(res["max_rowsum_err"] < 1e-4),
        "mean_entropy_ratio": float(ent_ratio.mean()),
        "min_head_entropy_ratio": float(ent_ratio.min()),
        "check_not_uniform (mean entropy ratio < 0.98)": bool(ent_ratio.mean() < 0.98),
        "top3_lags_minutes_by_rollout": [int(lags_min[i]) for i in top_lags],
        "last_step_attention_share": float(res["rollout"][-1] / res["rollout"].sum()),
        "high_vs_low_attention_l1_gap": float(np.abs(avg_layers[hi].mean(0) - avg_layers[lo].mean(0)).sum()),
        "permutation": perm_info,
    }
    save_json(summary, out / "attention_summary.json")
    print(summary)
    print(f"[OK] đã lưu vào {out}")


if __name__ == "__main__":
    main()
