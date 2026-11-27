"""Sinh dữ liệu GIẢ LẬP cùng schema với UCI Appliances Energy Prediction.

CHỈ dùng để smoke-test pipeline khi chưa có dữ liệu thật. KHÔNG dùng để báo cáo kết quả.
Ghi ra data/synthetic/energydata_synthetic.csv (không đụng vào data/raw/).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(n=19735, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2016-01-11 17:00:00", periods=n, freq="10min")
    hod = t.hour.to_numpy() + t.minute.to_numpy() / 60
    dow = t.dayofweek.to_numpy()
    day = np.arange(n) / 144.0

    t_out = 3 + 0.08 * day + 4 * np.sin(2 * np.pi * (hod - 9) / 24) + np.cumsum(rng.normal(0, 0.05, n)) * 0.3
    rh_out = np.clip(80 - 2.0 * (t_out - 5) + rng.normal(0, 4, n), 20, 100)
    occupancy = ((hod > 6) & (hod < 23)).astype(float) * (1.0 + 0.3 * (dow >= 5))
    lights = np.where(rng.random(n) < 0.15 * occupancy, rng.choice([10, 20, 30, 40], n), 0)

    base = 45 + 55 * np.exp(-0.5 * ((hod - 18.5) / 2.5) ** 2) + 30 * np.exp(-0.5 * ((hod - 8) / 1.5) ** 2)
    spikes = rng.random(n) < 0.04
    app = base * occupancy.clip(0.4, 1.4) * rng.lognormal(0, 0.35, n) + 0.8 * lights
    app = app + spikes * rng.lognormal(5.0, 0.6, n)
    app = np.clip(np.round(app / 10) * 10, 10, 1080)

    cols = {"date": t.strftime("%Y-%m-%d %H:%M:%S"), "Appliances": app.astype(int), "lights": lights.astype(int)}
    for i in range(1, 10):
        if i == 6:
            temp = t_out + rng.normal(0, 0.3, n)
            rh = np.clip(rh_out - 20 + rng.normal(0, 3, n), 1, 100)
        elif i == 9:
            temp = 19 + 0.002 * app + rng.normal(0, 0.4, n)
            rh = np.clip(40 + rng.normal(0, 3, n), 1, 100)
        else:
            temp = 19.5 + 1.5 * np.sin(2 * np.pi * (hod - 15) / 24) + 0.003 * app + 0.04 * (t_out - 5) + rng.normal(0, 0.3, n)
            rh = np.clip(40 - 0.6 * (temp - 20) + 0.003 * app + rng.normal(0, 2.5, n), 1, 100)
        cols[f"T{i}"] = np.round(temp, 2)
        cols[f"RH_{i}"] = np.round(rh, 2)
    cols["T_out"] = np.round(t_out, 2)
    cols["Press_mm_hg"] = np.round(755 + np.cumsum(rng.normal(0, 0.1, n)) * 0.5, 1)
    cols["RH_out"] = np.round(rh_out, 1)
    cols["Windspeed"] = np.round(np.clip(4 + rng.normal(0, 2, n), 0, 14), 1)
    cols["Visibility"] = np.round(np.clip(38 + rng.normal(0, 12, n), 1, 66), 1)
    cols["Tdewpoint"] = np.round(t_out - (100 - rh_out) / 5, 1)
    cols["rv1"] = np.round(rng.uniform(0, 50, n), 6)
    cols["rv2"] = cols["rv1"].copy()

    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "energydata_synthetic.csv"
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"[OK] ghi {path} ({n} dòng) - DỮ LIỆU GIẢ LẬP, chỉ để smoke test")


if __name__ == "__main__":
    main()
