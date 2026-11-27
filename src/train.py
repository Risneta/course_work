"""Huấn luyện Transformer / LSTM trên cửa sổ đã preprocess.

Chạy:
    python -m src.train --model all
    python -m src.train --model transformer --seeds 0
    python -m src.train --model all --set train.epochs=2        # chạy nhanh để kiểm tra
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import add_common_args, config_from_args, p
from .data import inverse_target
from .models import build_model, count_params
from .utils import ensure_dir, load_json, save_json, set_seed


def load_split(cfg, name):
    z = np.load(p(cfg, "processed_dir", f"{name}.npz"))
    return {k: z[k] for k in z.files}


def get_device(cfg):
    dev = cfg["train"]["device"]
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def rmse_wh(split, pred_scaled, sc):
    pred = inverse_target(pred_scaled, sc)
    return float(np.sqrt(np.mean((pred - split["y_raw"]) ** 2)))


def train_one(model_name, seed, cfg, data, sc, device):
    tc = cfg["train"]
    set_seed(seed)
    n_features = data["train"]["X"].shape[2]
    model = build_model(model_name, cfg, n_features).to(device)
    n_params = count_params(model)

    out_dir = ensure_dir(p(cfg, "output_dir", "models", model_name, f"seed{seed}"))
    (out_dir / "train_summary.json").unlink(missing_ok=True)  # xoá bản cũ để monitor biết run này đang chạy
    ds =TensorDataset(torch.from_numpy(data["train"]["X"]).float(), torch.from_numpy(data["train"]["y"]).float())
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, generator=gen,
                        num_workers=tc["num_workers"])

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=max(2, tc["patience"] // 3))
    loss_fn = nn.MSELoss() if tc["loss"] == "mse" else nn.SmoothL1Loss()

    best_val, best_epoch, best_state, bad = float("inf"), -1, None, 0
    history = []
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if tc["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        val_rmse = rmse_wh(data["val"], predict(model, data["val"]["X"], device), sc)
        sched.step(val_rmse)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tot / n, "val_rmse_wh": val_rmse, "lr": lr,
                        "elapsed_s": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)  # ghi mỗi epoch để monitor đọc được
        improved = val_rmse < best_val - 1e-6
        if improved:
            best_val, best_epoch, bad = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"[{model_name} seed{seed}] ep {epoch:3d} train_loss={tot / n:.4f} val_RMSE={val_rmse:.2f} Wh "
              f"lr={lr:.1e}{' *' if improved else ''}")
        if bad >= tc["patience"]:
            print(f"[{model_name} seed{seed}] early stop (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    cfg_key = model_name
    torch.save({"state_dict": best_state, "model_name": model_name, "model_cfg": cfg[cfg_key],
                "n_features": n_features, "seed": seed, "best_epoch": best_epoch,
                "best_val_rmse_wh": best_val}, out_dir / "best.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"model": model_name, "seed": seed, "n_params": n_params, "best_epoch": best_epoch,
               "epochs_run": len(history), "best_val_rmse_wh": best_val,
               "train_seconds": round(time.time() - t0, 1), "device": str(device)}
    save_json(summary, out_dir / "train_summary.json")
    save_json(cfg, out_dir / "config_snapshot.json")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--model", choices=["transformer", "lstm", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    seeds = args.seeds if args.seeds else cfg["train"]["seeds"]
    names = ["transformer", "lstm"] if args.model == "all" else [args.model]

    data = {s: load_split(cfg, s) for s in ("train", "val")}
    sc = load_json(p(cfg, "processed_dir", "scaler.json"))
    device = get_device(cfg)
    print(f"device={device} | train windows={len(data['train']['X'])} | features={data['train']['X'].shape[2]}")

    rows = []
    for name in names:
        for seed in seeds:
            rows.append(train_one(name, seed, cfg, data, sc, device))
    df = pd.DataFrame(rows)
    ensure_dir(p(cfg, "output_dir", "models"))
    df.to_csv(p(cfg, "output_dir", "models", "train_summary_all.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
