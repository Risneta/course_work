# 12 — Runbook: chạy end-to-end

## 1. Purpose
Một chỗ duy nhất có đầy đủ lệnh chạy, kết quả mong đợi ở từng bước, cách chạy nhanh để kiểm tra, và cách xử lý lỗi thường gặp.

## 2. IO
- **Input:** `data/raw/energydata_complete.csv`, `configs/config.yaml`, môi trường Python với `requirements.txt`.
- **Output:** toàn bộ cây `data/processed/` và `outputs/` theo `02_io_conventions.md`; kết quả cuối ở `outputs/results_report.md` và `outputs/attention/`.

### Bảng lệnh và kết quả mong đợi
Chạy từ thư mục `energy_ts/`, trong môi trường Python đã cài `requirements.txt`. Mọi lệnh `python -m src.<tên>` đều dùng được trên Windows, macOS và Linux.
| Bước | Lệnh | Doc | Kết quả mong đợi |
|---|---|---|---|
| 0 | `python -m pip install -r requirements.txt` | — | cài xong |
| 0b | `python scripts/check_structure.py` | 02 | “Cấu trúc và phiên bản code khớp với tài liệu” |
| 1 | `python -m pytest -q tests` | 05, 07 | tất cả pass (test model chỉ chạy khi có torch) |
| 2 | `python -m src.eda` | 04 | in EDA summary; sinh `outputs/eda/*` |
| 3 | `python -m src.preprocess` | 05 | `[OK] windows: {...}`, sinh `data/processed/*` |
| 4 | `python -m src.baselines` | 06 | bảng RMSE/MAE/R² của 5 baseline |
| 5 | `python -m src.train --model all` | 08 | log mỗi epoch; `best.pt` cho 2 mô hình × 3 seed |
| 5b | (terminal thứ hai, lúc train) `python -m src.monitor --watch 10` | 08 | bảng + `training_monitor.png` cập nhật mỗi 10 giây |
| 6 | `python -m src.evaluate` | 09 | `outputs/results_report.md`, PASS/FAIL |
| 7 | `python -m src.attention --seed 0` | 10 | hình attention + `attention_summary.json` |
| 8 | `python -m src.training_report` | 13 | `training_overview.png`, `training_per_run.png`, `training_runs_table.csv` |
| 9 | mở `notebooks/report.ipynb` → Run All | 13 | notebook trình bày đầy đủ |
| 10 | (tuỳ chọn) `scripts\run_ablation_no_temp.ps1` | 14 | `outputs_no_temp/ablation_comparison.md` |

**Chạy gộp:**
- Windows: `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1` (thêm `-Quick` để chạy nhanh).
- macOS/Linux: `bash scripts/run_all.sh` (thêm `QUICK=1` phía trước để chạy nhanh).
- Chạy nhanh = 2 epoch, 1 seed, chỉ để kiểm tra pipeline; **kết quả không có ý nghĩa khoa học**.

### Trạng thái kiểm thử (trung thực, cập nhật 26/11/2026)
| Phần | Trạng thái | Ghi chú |
|---|---|---|
| `config`, `data`, `metrics`, `eda`, `preprocess`, `baselines`, `tests/test_data.py` | Đã chạy cả trên dữ liệu giả lập (sandbox) lẫn dữ liệu thật (máy bạn) | EDA và baseline có kết quả thật (xem `04`, `06`) |
| `models`, `train`, `evaluate`, `attention` | **Đã chạy trên máy bạn, có kết quả thật** (xem `09`, `10`) | Sandbox của mình không có PyTorch nên mình chưa tự chạy được; kết quả dựa trên file bạn gửi |
| `tests/test_models.py` | Chưa xác nhận kết quả trên máy bạn | Cần torch; chạy `python -m pytest -q tests` |
| `monitor`, `training_report`, `notebooks/report.ipynb`, `compare_ablation`, `check_structure.py` | Mới chạy thử trên **dữ liệu giả lập** | Chưa chạy trên `outputs/` thật của bạn |
| `run_ablation_no_temp.ps1`, `run_all.ps1` | **Chưa chạy được ở đâu** (sandbox không có PowerShell); `.sh` đã kiểm tra cú pháp | Chạy `-Quick` trước |
| Ablation D21 trên dữ liệu thật | **Chưa chạy** | Xem `14` |

→ Có lỗi ở phần “chưa xác nhận” thì gửi log để sửa.

## 3. Success Criteria
- `scripts\run_all.ps1` (hoặc `bash scripts/run_all.sh`) kết thúc không lỗi.
- `python scripts/check_structure.py` không báo lỗi.
- Các file ở `02_io_conventions.md` mục 2.2 đều tồn tại.
- `outputs/results_report.md` có bảng kết quả, bảng bootstrap và danh sách PASS/FAIL.
- `outputs/attention/attention_summary.json` có `check_rows_sum_to_1: true`.
- Chạy lại với cùng seed cho kết quả val RMSE tương đương (xem `08`, T6).

## 4. How to do
1. Tạo môi trường. Windows: `python -m venv .venv`, `.venv\\Scripts\\activate`, `python -m pip install -r requirements.txt`. macOS/Linux: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
2. Đặt dataset vào `data/raw/` (xem `03`).
3. Chạy `python scripts/check_structure.py`, rồi chạy nhanh (`scripts\run_all.ps1 -Quick` hoặc `QUICK=1 bash scripts/run_all.sh`) để chắc pipeline thông suốt.
4. Chạy `python -m src.eda`, đọc `outputs/eda/eda_summary.md`, **bàn các quyết định mở** ở `11_decisions_log.md`; cập nhật config.
5. Chạy đầy đủ (`scripts\run_all.ps1` hoặc `bash scripts/run_all.sh`).
6. Đọc `outputs/results_report.md` và `outputs/attention/*`, viết nhận xét (xem `09`, `10`); xem tổng quan bằng `notebooks/report.ipynb` (xem `13`).

### Lỗi thường gặp
| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError ... energydata_complete.csv` | Chưa đặt dataset | Làm theo `03` |
| `ModuleNotFoundError: torch` | Chưa cài torch | `pip install torch` |
| `ModuleNotFoundError: src` | Chạy sai thư mục | Chạy từ thư mục `energy_ts/` bằng `python -m src.<tên>` |
| `attempted relative import with no known parent package` | Chạy trực tiếp file (`python src\\eda.py` hoặc nút Run) | Dùng `python -m src.<tên>` |
| `ModuleNotFoundError: yaml` (hoặc thư viện khác) | Chưa cài hoặc cài vào Python khác | `python -m pip install -r requirements.txt` (dùng `python -m pip`) |
| `ImportError: cannot import name 'training_report' from 'src'` | Thư mục `src` thiếu file mới | Chép lại từ zip mới nhất; chạy `python scripts/check_structure.py` |
| Ablation cho kết quả y hệt bản chính | `data.py` là bản cũ, `exclude_features` bị bỏ qua | Chép `src/data.py` mới; kiểm tra `data/processed_no_temp/feature_names.json` phải có 20 feature |
| `Chưa có checkpoint nào` ở `evaluate` | Chưa `train` hoặc `train.seeds` không khớp | Chạy `train` cho đúng các seed |
| `FileNotFoundError ... baselines/test_predictions.npz` | Chưa chạy `baselines` | Chạy bước 4 |
| `LinAlgWarning` từ Ridge | Feature tương quan cao | Bỏ qua |
| Val RMSE không giảm | `lr` quá cao/thấp hoặc dữ liệu chưa scale | Xem `08` mục 4.5 |
| Thiếu bộ nhớ GPU | batch quá lớn | `--set train.batch_size=64` |
