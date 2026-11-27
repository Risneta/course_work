# Multivariate Time-Series Regression — UCI Appliances Energy Prediction

Dự đoán mức tiêu thụ điện `Appliances` (Wh) **1 giờ sau** từ 6 giờ dữ liệu đa biến quá khứ.
Mô hình chính: **Transformer Encoder** · Baseline: **LSTM** (+ persistence, moving average, Ridge) · Extension: **phân tích attention map**.

**Bắt đầu từ `doc/00_index.md`** (15 file MD: Purpose / IO / Success Criteria / How to do) — ở đó có cây thư mục đầy đủ.

```powershell
python -m pip install -r requirements.txt
# đặt energydata_complete.csv vào data\raw\  (xem doc/03_data_setup.md)
python scripts/check_structure.py                      # kiểm tra cấu trúc và phiên bản code
python -m src.eda                                      # rồi đọc outputs\eda\eda_summary.md (doc/04, doc/11)
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Quick   # chạy thử nhanh
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1          # chạy đầy đủ
```
macOS/Linux: `bash scripts/run_all.sh` (thêm `QUICK=1` để chạy nhanh).

Sau khi chạy: `outputs/results_report.md` (kết quả + PASS/FAIL), `notebooks/report.ipynb` (trình bày), `doc/14_ablation_no_temperature.md` (ablation nhiệt độ).
