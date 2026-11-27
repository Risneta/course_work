# 02 — I/O conventions: mọi thứ lưu ở đâu

## 1. Purpose
Quy định **cố định** vị trí, tên, định dạng của mọi file input/output để các bước nối với nhau mà không cần sửa code. Đường dẫn gốc cấu hình trong `configs/config.yaml` mục `paths`; tất cả đường dẫn dưới đây tính từ thư mục `energy_ts/`. File này cũng mô tả hạ tầng dùng chung: `src/config.py` (đọc config, `--set`, hàm đường dẫn `p()`) và `src/utils.py` (tạo thư mục, đọc/ghi JSON, set seed).

## 2. IO

### 2.1 Input do người dùng cung cấp
| File | Định dạng | Ghi chú |
|---|---|---|
| `data/raw/energydata_complete.csv` | CSV, 29 cột, 19 735 dòng | Tải từ UCI, **không sửa** |

### 2.2 Registry: bước nào sinh / đọc file nào
| Bước | Sinh ra | Đọc vào |
|---|---|---|
| EDA | `outputs/eda/eda_summary.md`, `eda_report.json`, `descriptive_stats.csv`, `target_correlations.csv`, `split_drift_zscore.csv`, `*.png` | `data/raw/*.csv` |
| Preprocess | `data/processed/{train,val,test}.npz`, `scaler.json`, `feature_names.json`, `split_info.json`, `preprocess_report.json`, `split_timeline.png` | `data/raw/*.csv` |
| Baselines | `outputs/baselines/metrics.json`, `summary.csv`, `test_predictions.npz` | `data/processed/*` |
| Train | `outputs/models/{model}/seed{S}/best.pt`, `history.csv` (ghi mỗi epoch), `train_summary.json`, `config_snapshot.json`; `outputs/models/train_summary_all.csv` | `data/processed/{train,val}.npz`, `scaler.json` |
| Monitor (lúc đang train) | `outputs/figures/training_monitor.png` (và in bảng ra terminal) | `outputs/models/*/seed*/history.csv`, `outputs/baselines/metrics.json` |
| Evaluate | `outputs/predictions/{model}_seed{S}_test.npz`; `outputs/metrics/per_seed_metrics.csv`, `summary.csv`, `comparison_bootstrap.json`, `success_checks.json`; `outputs/figures/{pred_vs_actual,scatter_pred_vs_actual,error_by_hour,training_curves}.png`; `outputs/results_report.md` | `best.pt`, `test.npz`, `val.npz`, `scaler.json`, `outputs/baselines/test_predictions.npz` |
| Attention | `outputs/attention/*.png`, `permutation_importance.csv`, `attention_summary.json` | `best.pt`, `test.npz`, `feature_names.json` |
| Training report | `outputs/figures/training_overview.png`, `training_per_run.png`, `outputs/models/training_runs_table.csv` | `outputs/models/*/seed*/history.csv`, `config_snapshot.json`, `train_summary.json` |
| Notebook | (không ghi file; có thể xuất HTML/PDF thủ công) | `outputs/*`, `data/processed/*`, `configs/config.yaml` |
| Ablation D21 | `data/processed_no_temp/*`, `outputs_no_temp/*` (cùng cấu trúc `outputs/`) | `data/raw/*.csv`, `configs/config.yaml` |
| Compare ablation | `outputs_no_temp/ablation_comparison.{md,csv,png}` | `outputs/` (bản đầy đủ) và `outputs_no_temp/` |

### 2.3 Schema các file quan trọng
**`data/processed/{split}.npz`** (mỗi split cùng cấu trúc):
| Key | Shape | Dtype | Ý nghĩa |
|---|---|---|---|
| `X` | (N, L, F) | float32 | Input đã scale (z-score theo train) |
| `y` | (N,) | float32 | Target đã scale |
| `y_raw` | (N,) | float32 | Target gốc, Wh (dùng để tính metric) |
| `hist_raw` | (N, L) | float32 | Lịch sử `Appliances` gốc trong window (cho baseline) |
| `daily_raw` | (N,) | float32 | `Appliances` cùng giờ hôm trước (NaN nếu không có) |
| `tgt_idx` | (N,) | int32 | Chỉ số hàng (trong CSV đã sort) của bước target |
| `tgt_time` | (N,) | datetime64[s] | Thời điểm của bước target |

**`data/processed/scaler.json`:** `x_mean`, `x_std` (theo thứ tự `feature_names.json`), `y_mean`, `y_std`, `target_transform`.
**`outputs/models/.../best.pt`:** dict gồm `state_dict`, `model_name`, `model_cfg`, `n_features`, `seed`, `best_epoch`, `best_val_rmse_wh`.
**`outputs/models/.../history.csv`:** cột `epoch, train_loss, val_rmse_wh, lr, elapsed_s` (một dòng mỗi epoch).
**`outputs/metrics/success_checks.json`:** danh sách `{id, pass, detail}`. **`comparison_bootstrap.json`:** `{cặp: {delta_rmse, ci95_low, ci95_high, significant}}`.

### 2.4 Quy tắc đặt tên & ghi file
- Tên file: chữ thường, `snake_case`; seed ghi dạng `seed0`, `seed1`, …
- Đơn vị **mọi metric báo cáo là Wh gốc**, không phải giá trị đã scale.
- Mỗi bước chỉ **ghi** vào thư mục của nó; không bước nào sửa `data/raw/`.
- Bản ablation luôn ghi vào `data/processed_no_temp/` và `outputs_no_temp/`, **không bao giờ** ghi đè `data/processed/` hay `outputs/` (đổi bằng `--set paths.processed_dir=... --set paths.output_dir=...`).
- Thư mục sinh ra có thể xoá và dựng lại bằng pipeline: `data/processed*/`, `data/synthetic/`, `outputs*/`. Chúng nằm trong `.gitignore`.
- Kết quả trên dữ liệu **giả lập** (`data/synthetic/`) chỉ để thử code, không dùng cho báo cáo.

### 2.5 Thư mục được phép có ở gốc project
`configs/`, `data/`, `doc/`, `notebooks/`, `src/`, `scripts/`, `tests/`, `outputs/`, `outputs_no_temp/` và các file `README.md`, `requirements.txt`, `.gitignore`. Cây đầy đủ ở `00_index.md` mục 2.4.

## 3. Success Criteria
- Sau khi chạy đủ pipeline, mọi file ở bảng 2.2 tồn tại và không rỗng.
- Không thư mục nào ngoài danh sách 2.5 ở gốc project.
- `X.shape[1:] == (L, F)` với `F == len(feature_names.json)`; không có NaN/inf.
- `python scripts/check_structure.py` không báo lỗi.
- Kiểm tra nhanh schema: `python -c "import numpy as np; z=np.load('data/processed/test.npz'); print({k:z[k].shape for k in z.files})"`.

## 4. How to do
1. Tạo/kiểm tra thư mục: `data/raw/` chứa CSV. Các thư mục còn lại code tự tạo.
2. Chạy từng bước theo thứ tự ở `00_index.md`; sau mỗi bước đối chiếu file sinh ra với bảng 2.2.
3. Muốn đổi vị trí lưu: sửa `paths.*` trong `configs/config.yaml` (không sửa code).
4. Muốn chạy thử không đụng kết quả thật: thêm `--set paths.output_dir=outputs_try --set paths.processed_dir=data/processed_try`.
5. Sau khi chép file mới hoặc đổi cấu trúc: `python scripts/check_structure.py`. Lệnh này báo thiếu file, code là bản cũ, doc thiếu mục, hoặc doc nhắc tới file không tồn tại.
