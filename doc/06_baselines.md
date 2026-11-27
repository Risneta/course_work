# 06 — Baselines (không dùng deep learning)

## 1. Purpose
Có các mốc tham chiếu đơn giản. Mô hình deep learning chỉ có giá trị khi vượt được chúng; nếu không, độ phức tạp thêm vào là vô nghĩa.

## 2. IO
- **Input:** `data/processed/{train,val,test}.npz`, `scaler.json`; `baselines.ridge_alphas`.
- **Output** (trong `outputs/baselines/`):
  - `metrics.json` — metric val & test của từng baseline (+ `ridge_alpha` đã chọn)
  - `summary.csv` — bảng test (RMSE, MAE, MAPE, sMAPE, R², n)
  - `test_predictions.npz` — dự đoán test (Wh) của từng baseline, dùng lại ở `evaluate`

Các baseline:
| Tên | Dự đoán `Appliances[t+H]` bằng |
|---|---|
| `persistence` | giá trị cuối của window `Appliances[t]` |
| `moving_avg_6` | trung bình 6 bước cuối (1 giờ) |
| `moving_avg_all` | trung bình cả window (6 giờ) |
| `daily_naive` | giá trị cùng giờ hôm trước (`t+H-144`) |
| `ridge` | Ridge trên window đã flatten `(L·F)`; chọn `alpha` theo RMSE val |

## 3. Success Criteria
- `metrics.json`, `summary.csv`, `test_predictions.npz` tồn tại; mọi giá trị hữu hạn.
- Mọi baseline đánh giá trên **cùng** tập test, đơn vị Wh.
- `persistence` khớp với số “persistence_preview_test” của EDA (xấp xỉ; EDA tính trên đoạn test theo chỉ số hàng).
- **Mốc cần vượt** cho mô hình DL: `persistence`, `moving_avg_6` và `ridge`; **Ridge là mốc bắt buộc (V3, xem `09`)** vì persistence có R² âm trên test nên là mốc quá dễ. Không vượt được Ridge vẫn phải báo cáo trung thực.
- Ridge chọn alpha **chỉ bằng val**, không nhìn test.

### Kết quả thực tế (test, 2 961 mẫu, đơn vị Wh; bạn chạy ngày 26/11/2026)
| Baseline | RMSE | MAE | R² |
|---|---|---|---|
| ridge | **81.72** | 49.12 | 0.191 |
| moving_avg_6 | 88.26 | 42.81 | 0.057 |
| moving_avg_all | 89.77 | 48.81 | 0.024 |
| persistence | 101.66 | 46.29 | −0.251 |
| daily_naive | 109.47 | 52.49 | −0.451 |

Nhận xét: Ridge tốt nhất theo RMSE nhưng MAE kém persistence và moving_avg_6 (dự đoán tốt hơn ở đỉnh, kém hơn ở mức thấp). Baseline mạnh nhất chỉ giải thích được ~19% phương sai.

## 4. How to do
1. Chạy xong `preprocess` (05).
2. Chạy `python -m src.baselines`.
3. Đọc bảng in ra; ghi lại baseline mạnh nhất vào `11_decisions_log.md` (làm mốc so sánh).
4. Cảnh báo `LinAlgWarning: ill-conditioned` từ Ridge là bình thường (các feature tương quan cao) và không ảnh hưởng kết luận.
