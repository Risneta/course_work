# 05 — Preprocessing

## 1. Purpose
Biến CSV thô thành tensor huấn luyện được, **không rò rỉ thông tin tương lai**: làm sạch nhẹ, tạo feature, chia theo thời gian, scale, và cắt cửa sổ trượt.

## 2. IO
- **Input:** `data/raw/energydata_complete.csv`; config `data.*`, `window.*`, `split.*`.
- **Output** (trong `data/processed/`):
  - `train.npz`, `val.npz`, `test.npz` (schema ở `02_io_conventions.md` mục 2.3)
  - `scaler.json`, `feature_names.json`, `split_info.json`, `preprocess_report.json`, `split_timeline.png`

## 3. Success Criteria
Đều được `assert` tự động trong `src/preprocess.py` và test trong `tests/test_data.py`:
| # | Kiểm tra | Kỳ vọng |
|---|---|---|
| P1 | Không NaN/inf trong `X`, `y` | đúng |
| P2 | `max(tgt_idx[train]) < train_end ≤ min(tgt_idx[val])`, tương tự val/test | đúng (chia theo thời gian, không chồng lấn target) |
| P3 | Trung bình các hàng train sau scale ≈ 0 | `< 1e-6` (scaler fit trên train) |
| P4 | Bước cuối window cách target đúng `H` bước | đúng |
| P5 | `X.shape == (N, L, F)`, `F == len(feature_names)` | đúng |
| P6 | Số window mỗi split | ≈ 70/15/15 tổng (hao hụt ~`L+H-1` hàng đầu) |
| P7 | Val/test **không** có trung bình = 0 sau scale | đúng (chứng tỏ không fit scaler trên chúng) |

## 4. How to do
1. **Đọc & sort** theo `date`, kiểm tra (xem `03`).
2. **Bỏ cột** `rv1`, `rv2` (`data.drop_cols`).
3. **Nội suy tuyến tính** nếu có ô thiếu (dataset gốc không thiếu, bước này thường không làm gì).
4. **Biến đổi target** (tuỳ chọn): `target_transform: log1p` → áp dụng cho cột `Appliances` (cả input lẫn target). Metric luôn quy về Wh gốc.
5. **Tạo feature thời gian:** `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`.
6. **Chọn feature input:** toàn bộ cột còn lại; `use_target_as_input: true` giữ lịch sử `Appliances` làm feature. `data.exclude_features` (mặc định rỗng) loại thêm cột khỏi input, dùng cho ablation (xem `14`).
7. **Chia theo thời gian:** `train_end = int(0.70·T)`, `val_end = int(0.85·T)` (theo chỉ số hàng).
8. **Fit scaler (z-score) chỉ trên các hàng train**; áp dụng cho toàn chuỗi. Lưu `scaler.json`.
9. **Cắt window trên toàn chuỗi:** window `i` dùng hàng `[i, i+L-1]`, target là hàng `i+L-1+H`. Gán window vào split theo **chỉ số hàng của target** (input của val/test được phép chứa hàng quá khứ thuộc split trước — đó là dữ liệu quá khứ hợp lệ).
10. **Chạy assert P1–P4**, lưu `.npz` + JSON, vẽ `split_timeline.png`.
11. Chạy: `python -m src.preprocess`, rồi mở `preprocess_report.json` kiểm tra số window và danh sách feature.
