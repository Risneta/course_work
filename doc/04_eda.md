# 04 — EDA: đánh giá dữ liệu

## 1. Purpose
Hiểu dữ liệu **trước khi** mô hình hoá, và biến những gì thấy được thành quyết định cụ thể (có log-transform không, có bỏ biến nào không, chia split ra sao, mốc baseline là bao nhiêu).

## 2. IO
- **Input:** `data/raw/energydata_complete.csv`; tham số `data.*`, `window.horizon`, `split.*` trong config.
- **Output** (trong `outputs/eda/`):
  - `eda_summary.md` — **bản tóm tắt tự sinh, đọc file này đầu tiên**; kèm danh sách “Phát hiện & gợi ý quyết định”.
  - `eda_report.json` — số liệu máy đọc được.
  - `descriptive_stats.csv`, `target_correlations.csv`, `split_drift_zscore.csv`.
  - Hình: `target_distribution.png`, `target_timeseries.png`, `daily_weekly_profile.png`, `acf_target.png`, `correlation_heatmap.png`, `target_correlation_top.png`, `feature_distributions.png`, `split_drift.png`.

## 3. Success Criteria
EDA “đạt” khi trả lời được bằng số liệu từ các file output:
| # | Câu hỏi | Nằm ở đâu | Quyết định phụ thuộc |
|---|---|---|---|
| E1 | Dữ liệu sạch không (thiếu/trùng/không đều)? | `eda_summary.md` dòng đầu | Có cần nội suy/làm sạch thêm |
| E2 | Target lệch thế nào (skew, outlier)? | `target_distribution.png`, `eda_report.json → target` | `target_transform` (D3), loss (D4) |
| E3 | Có chu kỳ ngày/tuần không? | `daily_weekly_profile.png` | Giữ feature giờ/thứ (D7) |
| E4 | Phụ thuộc thời gian dài bao nhiêu? | `acf_target.png` | Chọn `lookback` (D6) |
| E5 | Feature nào liên quan đến target ở bước t+H? | `target_correlations.csv` | Giữ/bỏ feature |
| E6 | `rv1`, `rv2` có vô dụng không? | `eda_report.json → noise_vars` | Giữ trong `drop_cols` |
| E7 | Train/val/test có lệch phân phối không? | `split_drift.png`, `eda_report.json → split_target` | Kỳ vọng kết quả test, giải thích sai số |
| E8 | Bài toán khó cỡ nào? | `persistence_preview_test` | Mốc tối thiểu mô hình phải vượt |

Quy tắc đọc kết quả: outlier theo IQR của target là **đỉnh tiêu thụ thật** → không xoá. Chỉ xoá/sửa khi chứng minh là lỗi cảm biến.

### Kết quả EDA thực tế (dữ liệu UCI thật, bạn chạy ngày 25/11/2026)
| Mục | Giá trị |
|---|---|
| Quy mô | 19 735 dòng, 2016-01-11 17:00 → 2016-05-27 18:00; thiếu 0 ô, trùng 0, bước không đều 0 |
| Target `Appliances` | mean 97.7, median 60.0, std 102.5, min 10, max 1080, P99 577, **skew 3.39** (sau log1p còn 1.19) |
| Outlier theo IQR | 10.8% điểm (đỉnh tiêu thụ thật, không xoá) |
| Chu kỳ ngày | cao điểm 18h, thấp nhất 3h (tỉ lệ 3.95×) |
| ACF | lag1 = 0.75; lag3 = 0.44; lag6 = 0.32; lag36 = 0.02; lag144 = 0.22 |
| Tương quan feature[t] với target[t+6] | cao nhất chỉ ~0.13: `RH_out` −0.13, `lights` 0.12, `T6` 0.12, `Windspeed` 0.09, `T_out` 0.09 |
| `lights` | bằng 0 ở 77% số dòng |
| `rv1`, `rv2` | không tương quan với target → bỏ |
| Persistence trên đoạn test | RMSE 101.8 Wh, R² −0.252 |
| Drift train → test (đơn vị std của train) | `T9` +3.14, `T5` +3.05, `T4` +2.89, `T3` +2.82, `T7` +2.80, `T2` +2.70, `T1` +2.55, `T6` +2.17, `T_out` +2.12, `T8` +1.91, `Tdewpoint` +1.78, `RH_6` −1.73; target chỉ −0.02 |

Kết luận từ EDA: (1) target lệch phải mạnh → D3 chọn `log1p`; (2) chỉ có phụ thuộc ngắn hạn (ACF lag36 ≈ 0) cộng chu kỳ ngày → `lookback = 36` cùng feature giờ/thứ; (3) tín hiệu dự đoán yếu (tương quan ≤ 0.13) nên trần R² thấp; (4) test ấm hơn train rõ rệt (nhiệt độ lệch +2 đến +3 std) trong khi target không lệch → D18 và ablation D21.

## 4. How to do
1. Đảm bảo `data/raw/energydata_complete.csv` đã có (xem `03`).
2. Chạy `python -m src.eda`.
3. Đọc `outputs/eda/eda_summary.md`, rồi xem lần lượt các hình theo bảng E1–E8.
4. Với mỗi phát hiện, ghi quyết định vào `11_decisions_log.md` (đặc biệt D3, D4, D6).
5. Trao đổi với nhau các quyết định mở trước khi chạy `preprocess` nếu chúng làm đổi `config.yaml`.
6. Lưu ý: EDA dùng **toàn bộ** dữ liệu để quan sát, nhưng mọi quyết định có fit tham số (scaler, chọn siêu tham số) chỉ được dựa trên **train** để tránh leakage (xem `05`).
