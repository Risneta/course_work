# 00 — Index: bản đồ tài liệu, pipeline và cấu trúc thư mục

## 1. Purpose
Điểm vào duy nhất của thư mục `doc/`. Cho biết: có những file MD nào (00–14), mỗi file mô tả code nào, thứ tự chạy, cây thư mục **đúng như hiện tại**, và mọi thứ lưu ở đâu. Khi cấu trúc thay đổi, file này và `02_io_conventions.md` phải được cập nhật; kiểm tra bằng `python scripts/check_structure.py`.

## 2. IO
- **Input:** không có (đây là tài liệu).
- **Output:** không sinh file. Đọc tiếp các file bên dưới.

### 2.1 Danh sách tài liệu (đọc/làm theo thứ tự số)
| # | File MD | Mô tả | Code liên quan | Lệnh |
|---|---|---|---|---|
| 01 | `01_overview.md` | Định nghĩa bài toán, tiêu chí thành công tổng thể | `configs/config.yaml` | — |
| 02 | `02_io_conventions.md` | **Quy định mọi file input/output nằm ở đâu**, schema, kiểm tra cấu trúc | `src/config.py`, `src/utils.py`, `scripts/check_structure.py` | `python scripts/check_structure.py` |
| 03 | `03_data_setup.md` | Tải & đặt dataset, kiểm tra | `src/data.py` | (bạn tải tay) |
| 04 | `04_eda.md` | Khám phá dữ liệu, rút quyết định; kết quả EDA thực tế | `src/eda.py` | `python -m src.eda` |
| 05 | `05_preprocessing.md` | Làm sạch, feature, chia split, scaler, window | `src/preprocess.py`, `src/data.py` | `python -m src.preprocess` |
| 06 | `06_baselines.md` | Persistence, moving average, daily naive, Ridge | `src/baselines.py` | `python -m src.baselines` |
| 07 | `07_models.md` | Kiến trúc Transformer Encoder và LSTM | `src/models.py` | — |
| 08 | `08_training.md` | Huấn luyện, early stopping, seed, **theo dõi training** | `src/train.py`, `src/monitor.py` | `python -m src.train` + `python -m src.monitor --watch 10` |
| 09 | `09_evaluation.md` | Metric, so sánh, PASS/FAIL; **kết quả thực tế** | `src/evaluate.py`, `src/metrics.py` | `python -m src.evaluate` |
| 10 | `10_attention_analysis.md` | Phân tích attention map + permutation importance | `src/attention.py` | `python -m src.attention --seed 0` |
| 11 | `11_decisions_log.md` | Quyết định đã chốt, quyết định mở, thí nghiệm chưa làm | — | — |
| 12 | `12_runbook.md` | Chạy end-to-end (Windows + bash), trạng thái kiểm thử, xử lý lỗi | `scripts/run_all.ps1`, `scripts/run_all.sh` | `scripts\run_all.ps1` |
| 13 | `13_notebook_report.md` | Notebook trình bày + biểu đồ toàn bộ quá trình train, lý do dừng sớm | `notebooks/report.ipynb`, `src/training_report.py` | `python -m src.training_report` |
| 14 | `14_ablation_no_temperature.md` | Ablation D21: bỏ nhóm nhiệt độ, so sánh, kiểm định | `src/compare_ablation.py`, `src/data.py`, `scripts/run_ablation_no_temp.*` | `scripts\run_ablation_no_temp.ps1` |

### 2.2 Bản đồ code → doc (mỗi file code có đúng một nơi mô tả)
| File | Mô tả trong |
|---|---|
| `src/config.py`, `src/utils.py` | `02` (hạ tầng: đọc config, đường dẫn, JSON, seed) |
| `src/data.py` | `03` (load/validate), `05` (feature, scaler, window), `14` (`exclude_features`) |
| `src/eda.py` | `04` |
| `src/preprocess.py` | `05` |
| `src/baselines.py` | `06` |
| `src/models.py` | `07` |
| `src/train.py`, `src/monitor.py` | `08` |
| `src/metrics.py`, `src/evaluate.py` | `09` |
| `src/attention.py` | `10` |
| `src/training_report.py`, `notebooks/report.ipynb` | `13` |
| `src/compare_ablation.py`, `scripts/run_ablation_no_temp.ps1`, `scripts/run_ablation_no_temp.sh` | `14` |
| `scripts/run_all.ps1`, `scripts/run_all.sh`, `scripts/make_synthetic.py`, `scripts/check_structure.py` | `12`, `03`, `02` |
| `tests/test_data.py`, `tests/test_models.py` | `05`, `07` (và `12`) |

### 2.3 Sơ đồ pipeline
```
data/raw/energydata_complete.csv
        │  (03)
        ├──► EDA (04) ──► outputs/eda/*   ──► quyết định (11)
        ▼
 preprocess (05) ──► data/processed/{train,val,test}.npz, scaler.json, ...
        ├──► baselines (06) ──► outputs/baselines/*
        ▼
 train (08) [models (07)] ──► outputs/models/{transformer,lstm}/seed*/{best.pt,history.csv,...}
        │         └── monitor (08): xem trực tiếp trong lúc train
        ▼
 evaluate (09) ──► outputs/metrics/*, outputs/predictions/*, outputs/figures/*, outputs/results_report.md
        ▼
 attention (10) ──► outputs/attention/*
        ▼
 training_report (13) ──► outputs/figures/training_*.png, outputs/models/training_runs_table.csv
        ▼
 notebooks/report.ipynb (13)

 Nhánh ablation (14): preprocess/baselines/train/evaluate với data.exclude_features
        ──► data/processed_no_temp/, outputs_no_temp/ ──► compare_ablation ──► outputs_no_temp/ablation_comparison.md
```

### 2.4 Cây thư mục project (đúng như hiện tại)
```
energy_ts/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml              # mọi tham số
├── data/
│   ├── raw/                     # INPUT: energydata_complete.csv (bạn đặt vào) + README.md
│   ├── processed/               # sinh bởi preprocess
│   ├── processed_no_temp/       # sinh bởi ablation D21 (tuỳ chọn)
│   └── synthetic/               # dữ liệu giả lập, CHỈ để smoke test (tuỳ chọn)
├── doc/                         # 15 file MD: 00_index ... 14_ablation_no_temperature
├── notebooks/
│   └── report.ipynb             # notebook trình bày
├── src/
│   ├── __init__.py
│   ├── config.py  utils.py  metrics.py  data.py      # hạ tầng, dữ liệu
│   ├── eda.py  preprocess.py  baselines.py           # EDA, tiền xử lý, baseline
│   ├── models.py  train.py  monitor.py               # mô hình, huấn luyện, theo dõi
│   ├── evaluate.py  attention.py                     # đánh giá, attention
│   └── training_report.py  compare_ablation.py       # báo cáo training, so sánh ablation
├── scripts/
│   ├── run_all.ps1  run_all.sh                       # chạy toàn bộ pipeline
│   ├── run_ablation_no_temp.ps1  run_ablation_no_temp.sh
│   ├── make_synthetic.py                             # dữ liệu giả lập để thử code
│   └── check_structure.py                            # kiểm tra cấu trúc + phiên bản code
├── tests/
│   ├── test_data.py  test_models.py
├── outputs/                     # kết quả bản chính (sinh ra)
│   ├── eda/  baselines/  models/  predictions/  metrics/  figures/  attention/
│   └── results_report.md
└── outputs_no_temp/             # kết quả ablation (cùng cấu trúc outputs/ + ablation_comparison.*)
```
Thư mục sinh ra (`data/processed*`, `data/synthetic`, `outputs*`) có thể xoá và sinh lại bằng pipeline.

## 3. Success Criteria
- Mỗi file trong `src/` và `scripts/` xuất hiện trong bảng 2.2; mỗi file MD có đủ 4 mục Purpose / IO / Success Criteria / How to do.
- Cây 2.4 khớp thư mục thật: `python scripts/check_structure.py` kết thúc với “Cấu trúc và phiên bản code khớp với tài liệu”.
- Người mới đọc `00` → `14` có thể chạy lại toàn bộ mà không cần hỏi thêm.

## 4. How to do
1. Đọc `01_overview.md` để nắm bài toán và tiêu chí chung.
2. Đọc `02_io_conventions.md` để biết file nào đặt ở đâu.
3. Chạy `python scripts/check_structure.py` để chắc code và doc cùng phiên bản (nếu báo “bản cũ”, chép lại từ zip mới nhất).
4. Làm theo `03` → `10` (mỗi file có lệnh riêng) hoặc chạy gộp theo `12_runbook.md`.
5. Sau EDA (`04`), đọc `outputs/eda/eda_summary.md` rồi quay lại `11_decisions_log.md` để chốt các quyết định mở.
6. Sau khi train: `13` (biểu đồ + notebook) và, nếu cần, `14` (ablation).
7. Khi đổi tham số, sửa `configs/config.yaml` (hoặc `--set key=value`) và ghi lại lý do vào `11_decisions_log.md`.
