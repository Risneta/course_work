# 09 — Evaluation: thế nào là “đúng”, kết quả ra gì

## 1. Purpose
Đo mô hình trên tập test một lần, bằng metric có ý nghĩa vật lý (Wh), so sánh Transformer ↔ LSTM ↔ baseline có kiểm định thống kê, và tự động kết luận PASS/FAIL theo các tiêu chí đã chốt trước.

## 2. IO
- **Input:** `outputs/models/{model}/seed*/best.pt`, `data/processed/{val,test}.npz`, `scaler.json`, `outputs/baselines/test_predictions.npz`.
- **Output:**
  - `outputs/predictions/{model}_seed{S}_test.npz` — `y_raw`, `pred`, `tgt_time`
  - `outputs/metrics/per_seed_metrics.csv` — metric val & test từng seed
  - `outputs/metrics/summary.csv` — bảng tổng hợp (baseline; deep mean±std qua seed; seed-ensemble)
  - `outputs/metrics/comparison_bootstrap.json` — ΔRMSE + CI95
  - `outputs/metrics/success_checks.json` — danh sách tiêu chí PASS/FAIL
  - `outputs/figures/`: `pred_vs_actual.png`, `scatter_pred_vs_actual.png`, `error_by_hour.png`, `training_curves.png`
  - `outputs/results_report.md` — **báo cáo kết quả tự sinh**

### Metric (mọi giá trị tính trên Wh gốc)
RMSE (chính), MAE, MAPE %, sMAPE %, R². `Appliances` luôn ≥ 10 Wh nên MAPE không chia cho 0.

### Kết quả thực tế (test, 2 961 mẫu, Wh; theo `results_report.md` bạn gửi ngày 26/11/2026)
Lưu ý: cấu hình `target_transform: log1p` chưa được xác nhận bằng `data/processed/scaler.json` của lần chạy này; hãy kiểm tra trước khi trích dẫn.

| model | #seed | #params | RMSE | ±std | MAE | MAPE % | R² |
|---|---|---|---|---|---|---|---|
| persistence | 1 | 0 | 101.66 | 0 | 46.29 | 38.1 | −0.251 |
| moving_avg_6 | 1 | 0 | 88.26 | 0 | 42.81 | 37.5 | 0.057 |
| ridge | 1 | 0 | **81.72** | 0 | 49.12 | 53.8 | 0.191 |
| transformer | 3 | 73 409 | 82.39 | 2.56 | 45.14 | 46.9 | 0.177 |
| transformer_seed_ensemble | 3 | 73 409 | **79.72** | — | 43.95 | 45.7 | 0.230 |
| lstm | 3 | 62 337 | 87.52 | 3.81 | 52.74 | 58.0 | 0.071 |
| lstm_seed_ensemble | 3 | 62 337 | 83.33 | — | 49.78 | 54.9 | 0.159 |

Bootstrap (ΔRMSE = a − b, âm nghĩa a tốt hơn; CI95 chứa 0 nghĩa chưa có bằng chứng):
| cặp | ΔRMSE | CI95 | ý nghĩa |
|---|---|---|---|
| transformer vs lstm (ensemble) | −3.61 | [−5.73, −1.34] | có |
| transformer ensemble vs persistence / moving_avg_6 | −21.94 / −8.54 | CI không chứa 0 | có |
| transformer ensemble vs ridge | −2.00 | [−4.70, +0.62] | không |
| transformer seed0 / seed1 / seed2 vs ridge | −0.99 / −1.30 / +4.28 | đều chứa 0 | không |
| lstm ensemble vs ridge | +1.61 | [−0.49, +4.28] | không |
| lstm ensemble vs moving_avg_6 | −4.92 | [−10.53, +1.15] | không |

PASS/FAIL: Transformer PASS V1, V2 (cả điểm ước lượng lẫn ý nghĩa thống kê), **FAIL V3 (Ridge)** (2/3 seed thắng, không có ý nghĩa); LSTM PASS V1, PASS V2 ở điểm ước lượng nhưng FAIL ở ý nghĩa thống kê, **FAIL V3** (0/3 seed thắng). V4 (CV 3.1% và 4.4%) và V5 (val 81.77 / test 82.39 và val 83.48 / test 87.52) PASS cho cả hai.

Diễn giải: Transformer hơn LSTM và hơn các baseline đơn giản, nhưng **chưa vượt được Ridge một cách chắc chắn**. R² cao nhất 0.23, nên phần lớn phương sai chưa giải thích được. MAE và MAPE của các mô hình không tốt hơn baseline đơn giản dù RMSE tốt hơn, nên báo cáo cả ba.

## 3. Success Criteria
Cách kiểm tra (trong `evaluate.py`), tiêu chí chọn **trước khi xem kết quả**:
| ID | Tiêu chí | PASS khi |
|---|---|---|
| V1 | `{model}_beats_persistence` | RMSE (mean qua seed) < RMSE persistence |
| V2 | `{model}_beats_moving_avg_6` | RMSE < RMSE moving_avg_6 |
| V3 | `{model}_beats_ridge` **(BẮT BUỘC)** | RMSE (mean qua seed) < RMSE Ridge. Ridge là mốc có ý nghĩa nhất vì persistence có R² âm trên test. FAIL vẫn phải báo cáo trung thực |
| V3b | `…_significant` (cho V1, V2, V3) | CI95 của ΔRMSE (ensemble vs baseline, block bootstrap) nằm hoàn toàn dưới 0. PASS ở điểm ước lượng nhưng FAIL ở đây nghĩa là “chưa có bằng chứng thắng” |
| V4 | `{model}_seed_stability` | CV = std/mean của RMSE qua seed < 10% |
| V5 | `{model}_no_severe_overfit_to_val` | RMSE test ≤ 1.5 × RMSE val |
| V6 | Kết luận Transformer vs LSTM | Chỉ nói “A tốt hơn B” khi CI95 của ΔRMSE (block bootstrap) **không chứa 0**; nếu chứa 0 → “chưa có bằng chứng khác biệt” |

Diễn giải: V1–V2 là điều kiện cần. V3 (bắt buộc) cho biết deep learning có đáng dùng so với Ridge hay không. Dòng `*_ensemble_vs_*` so ensemble 3 seed với một baseline đơn nên có lợi cho ensemble; dòng `*_seedS_vs_ridge` so từng mô hình đơn lẻ, công bằng hơn. Báo cáo nên nêu cả hai. V6 ngăn việc kết luận dựa trên chênh lệch do ngẫu nhiên.

## 4. How to do
1. Hoàn tất `baselines` (06) và `train` (08) cho đủ seed trong `train.seeds`.
2. Chạy `python -m src.evaluate`.
3. Mở `outputs/results_report.md`: đọc bảng, rồi các dòng PASS/FAIL.
4. Xem `pred_vs_actual.png` và `error_by_hour.png` để hiểu **mô hình sai ở đâu** (thường là các đỉnh đột biến và khung giờ cao điểm), không chỉ nhìn một con số RMSE.
5. Khi viết báo cáo: nêu RMSE ± std qua seed, ΔRMSE kèm CI95, và nêu rõ các tiêu chí FAIL (nếu có) cùng nguyên nhân khả dĩ (drift, đỉnh ngẫu nhiên, nhiễu cảm biến).
6. Chỉ chạy `evaluate` để **báo cáo**, không dùng test để chỉnh siêu tham số. Nếu lỡ chỉnh theo test, nói rõ trong báo cáo.
