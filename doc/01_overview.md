# 01 — Overview: định nghĩa bài toán

## 1. Purpose
Chốt chính xác bài toán để mọi file sau dùng chung một định nghĩa: **hồi quy chuỗi thời gian đa biến** dự đoán mức tiêu thụ điện của thiết bị gia dụng từ các chỉ số quá khứ; dùng **Transformer Encoder**, so sánh với **LSTM baseline**, và phân tích **attention map**.

## 2. IO
**Dataset:** UCI *Appliances Energy Prediction* — `energydata_complete.csv`
- 19 735 dòng, 1 dòng / 10 phút, từ 2016-01-11 17:00 đến 2016-05-27 18:00.
- Cột: `date`, `Appliances` (target, Wh), `lights` (Wh), `T1..T9` / `RH_1..RH_9` (nhiệt độ, độ ẩm trong nhà), `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `rv1`, `rv2` (hai biến ngẫu nhiên).

**Công thức bài toán (theo `configs/config.yaml`):**
- Input mỗi mẫu: ma trận `(L=36, F)` — 36 bước quá khứ (6 giờ) × F feature (khoảng 31 sau xử lý, gồm cả lịch sử `Appliances` và feature thời gian).
- Output: một số thực — `Appliances` tại bước `t + H`, với `H = 6` (1 giờ sau bước cuối của window), đơn vị **Wh**.
- Chia train/val/test = 70/15/15 **theo thời gian** (không shuffle).
- Mô hình: `TransformerRegressor` (chính), `LSTMRegressor` (baseline DL), thêm baseline đơn giản: persistence, moving average, Ridge.
- Metric chính: **RMSE (Wh)** trên test; phụ: MAE, MAPE, sMAPE, R².

## 3. Success Criteria
Tiêu chí tổng thể (kiểm tra tự động trong `outputs/metrics/success_checks.json`, chi tiết ở `09`):

| ID | Tiêu chí | Cách kiểm |
|---|---|---|
| S1 | Pipeline chạy hết từ raw → attention, không lỗi | `run_all.sh` kết thúc exit 0 |
| S2 | Không rò rỉ dữ liệu (scaler fit trên train; target val/test sau train) | assert trong `preprocess.py` + `tests/` |
| S3 | Transformer và LSTM có RMSE test **thấp hơn persistence, moving_avg_6 và Ridge** (Ridge là mốc bắt buộc, V3), kèm kiểm tra ý nghĩa thống kê (V3b) | `success_checks.json` |
| S4 | Kết quả ổn định qua 3 seed (CV của RMSE < 10%) | `success_checks.json` |
| S5 | So sánh Transformer vs LSTM kèm khoảng tin cậy (block bootstrap), không kết luận khi CI chứa 0 | `comparison_bootstrap.json` |
| S6 | Attention hợp lệ (mỗi hàng tổng = 1, không đều tuyệt đối) và có đối chiếu với permutation importance | `attention_summary.json` |

Ghi chú trung thực: S3 là kỳ vọng, **không phải bảo đảm**. Dataset này nổi tiếng khó (nhiễu cao, nhiều đỉnh đột biến). Nếu mô hình không thắng baseline thì báo cáo đúng như vậy và phân tích nguyên nhân (drift, đỉnh bất thường, v.v.) — đó vẫn là kết quả hợp lệ.

## 4. How to do
1. Đọc kỹ mục IO để thống nhất `L`, `H`, target, đơn vị.
2. Mọi thay đổi `L`, `H`, split, transform đều làm trong `configs/config.yaml`, rồi chạy lại từ `preprocess`.
3. Không đổi định nghĩa bài toán giữa chừng khi đang so sánh mô hình (mọi mô hình phải dùng cùng `train/val/test.npz`).
4. Sau khi có kết quả, đối chiếu từng tiêu chí S1–S6 và ghi PASS/FAIL vào báo cáo (tự sinh ở `outputs/results_report.md`).
