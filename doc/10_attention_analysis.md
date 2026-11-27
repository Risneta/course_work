# 10 — Attention analysis (phần Extension)

## 1. Purpose
Xem Transformer “nhìn” vào những bước thời gian nào khi dự đoán, head nào chuyên biệt hoá, và đối chiếu với một phương pháp độc lập (permutation importance, kể cả cho LSTM) vì **attention không tự động là lời giải thích nhân quả**.

## 2. IO
- **Input:** `outputs/models/transformer/seed{S}/best.pt` (mặc định `--seed 0`), `outputs/models/lstm/seed{S}/best.pt` (cho permutation), `data/processed/test.npz`, `feature_names.json`, `scaler.json`.
- **Output** (trong `outputs/attention/`):
  | File | Nội dung |
  |---|---|
  | `attn_heads_over_time.png` | Attention từ token CLS tới từng bước thời gian, theo (layer, head), trung bình trên test |
  | `attn_by_lag.png` | Attention theo độ trễ (phút) mỗi layer + attention rollout, so với mức đều 1/L |
  | `attn_full_matrix.png` | Ma trận attention đầy đủ (query × key) trung bình mỗi layer |
  | `attn_entropy.png` | Entropy chuẩn hoá của từng head (1 = đều hoàn toàn) |
  | `attn_high_vs_low.png` | So attention giữa mẫu tiêu thụ cao (≥ P90) và thấp (< median) |
  | `attn_examples.png` | Vài mẫu riêng lẻ: attention chồng lên chuỗi `Appliances` |
  | `permutation_importance.csv/.png` | Tăng RMSE (Wh) khi xáo trộn từng feature, Transformer vs LSTM |
  | `attention_summary.json` | Số liệu tóm tắt và các kiểm tra tự động |

## 3. Success Criteria
| # | Kiểm tra | Kỳ vọng | Ở đâu |
|---|---|---|---|
| A1 | Mỗi hàng attention cộng = 1 | `max_rowsum_error < 1e-4` | `attention_summary.json` |
| A2 | Attention không đều tuyệt đối | `mean_entropy_ratio < 0.98` | `attention_summary.json` |
| A3 | Có thể nêu được ≥ 1 quan sát cụ thể bằng số: các lag nhận nhiều attention nhất, độ chuyên biệt của head, khác biệt cao/thấp | `top3_lags_minutes_by_rollout`, `high_vs_low_attention_l1_gap` | `attention_summary.json` |
| A4 | Có đối chiếu: feature quan trọng theo permutation có hợp lý với EDA (vd. lịch sử `Appliances`, `lights`, giờ trong ngày, nhiệt/ẩm ngoài trời) | so `permutation_importance.csv` với `target_correlations.csv` | báo cáo |
| A5 | Diễn giải **thận trọng**: không khẳng định nhân quả chỉ từ attention; nêu giới hạn | có đoạn “Giới hạn” trong báo cáo | báo cáo |

Cách đọc kết quả: nếu rollout dồn vào vài bước cuối → mô hình chủ yếu dùng thông tin gần; nếu có đỉnh lặp lại ở lag xa (vd. quanh −60 phút) → có thể có phụ thuộc theo chu kỳ ngắn; nếu entropy gần 1 ở mọi head → attention gần như đều, mô hình có thể không dùng cấu trúc thời gian nhiều. Đây là giả thuyết để kiểm tra, **không phải kết luận có sẵn**.

### Kết quả thực tế (Transformer seed 0, bạn chạy ngày 26/11/2026)
| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| `max_rowsum_error` | 3.6e-7 | A1 đạt: mỗi hàng attention cộng bằng 1 |
| `mean_entropy_ratio` / `min_head_entropy_ratio` | 0.933 / 0.815 | A2 đạt (< 0.98) nhưng attention **khá phân tán**, cấu trúc mờ |
| `top3_lags_minutes_by_rollout` | −10, −20, −70 phút | −10 và −20 hợp lý (thông tin gần nhất); −70 chưa giải thích được từ số, cần xem `attn_by_lag.png` |
| `last_step_attention_share` | 0.048 | bước cuối nhận khoảng 4.8% (mức đều = 1/36 ≈ 2.8%) |
| `base_rmse` permutation seed 0 | Transformer 80.74; LSTM 92.52 | LSTM seed 0 tệ hơn trung bình LSTM (87.52) nên không dùng để kết luận về LSTM |

Chưa có nhận xét cho từng hình attention và `permutation_importance.csv` vì chưa được gửi; A4 chưa hoàn thành.

## 4. How to do
1. Cần checkpoint từ bước `train` (08), tốt nhất sau khi đã `evaluate`.
2. Chạy `python -m src.attention --seed 0` (thêm `--skip-permutation` nếu muốn bỏ phần permutation cho nhanh).
3. Kiểm tra A1–A2 trong `attention_summary.json` trước; nếu A1 hỏng thì hình không đáng tin.
4. Xem lần lượt: `attn_by_lag.png` → `attn_heads_over_time.png` → `attn_entropy.png` → `attn_high_vs_low.png` → `attn_examples.png`.
5. So sánh với `permutation_importance.png`; viết nhận xét trong báo cáo kèm đoạn Giới hạn (attention ≠ giải thích; một seed; test nhỏ).
6. Để chắc chắn hơn, lặp lại với `--seed 1`, `--seed 2` và xem hình có nhất quán không (nên nêu trong báo cáo).
