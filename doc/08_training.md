# 08 — Training

## 1. Purpose
Huấn luyện Transformer và LSTM với **cùng** dữ liệu, loss, optimizer, lịch dừng và nhiều seed, để kết quả so sánh đáng tin.

## 2. IO
- **Input:** `data/processed/{train,val}.npz`, `scaler.json`; config `train.*`, `transformer.*`, `lstm.*`. (Test **không** được đọc ở bước này.)
- **Output** (cho mỗi `model ∈ {transformer, lstm}` và `seed`), trong `outputs/models/{model}/seed{S}/`:
  - `best.pt` — checkpoint ở epoch có val RMSE thấp nhất
  - `history.csv` — `epoch, train_loss, val_rmse_wh, lr, elapsed_s` (ghi **sau mỗi epoch**)
  - `train_summary.json` — `n_params, best_epoch, epochs_run, best_val_rmse_wh, train_seconds, device`
  - `config_snapshot.json` — config lúc chạy
  - tổng hợp: `outputs/models/train_summary_all.csv`
  - `outputs/figures/training_monitor.png` do `src/monitor.py` sinh ra; biểu đồ đầy đủ mọi run và lý do dừng do `src/training_report.py` sinh ra (xem `13`)

Thiết lập: AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch 128, loss MSE trên target đã scale (có thể đổi `huber`), gradient clipping 1.0, `ReduceLROnPlateau` theo val RMSE, **early stopping theo val RMSE tính bằng Wh** (`patience=12`), tối đa 100 epoch, seeds `[0,1,2]`.

## 3. Success Criteria
| # | Kiểm tra | Kỳ vọng |
|---|---|---|
| T1 | Có đủ `best.pt` cho mọi (model, seed) | đúng |
| T2 | Train loss giảm trong những epoch đầu | đúng; nếu không → kiểm tra `lr`, scale dữ liệu |
| T3 | Val RMSE tốt nhất < val RMSE của epoch 1 | đúng (mô hình có học) |
| T4 | Early stopping kích hoạt trước `epochs` hoặc đạt cực tiểu rõ ràng | `best_epoch < epochs_run` hoặc đường cong phẳng |
| T5 | Không có NaN trong `history.csv` | đúng |
| T6 | Tái lập: chạy lại cùng seed cho val RMSE tương đương | CPU: gần như trùng; GPU: sai khác nhỏ do thuật toán không tất định |
| T7 | Không có dòng nào đọc `test.npz` trong `train.py` | đúng (chống chọn mô hình theo test) |

### Theo dõi training (`src/monitor.py`)
Mở **terminal thứ hai** (cùng thư mục `energy_ts/`, cùng môi trường Python) trong lúc `train` đang chạy:
| Lệnh | Tác dụng |
|---|---|
| `python -m src.monitor` | In bảng trạng thái từng run + lưu `outputs/figures/training_monitor.png` (1 lần) |
| `python -m src.monitor --watch 10` | Làm mới bảng và hình mỗi 10 giây (Ctrl+C để dừng) |
| `python -m src.monitor --live` | Mở cửa sổ biểu đồ cập nhật trực tiếp (cần máy có giao diện đồ hoạ) |

Bảng gồm: epoch hiện tại, train loss, val RMSE, val RMSE tốt nhất và epoch đạt được, learning rate, thời gian đã chạy, trạng thái (`đang chạy (còn N epoch tới early-stop)` hoặc `xong`). Đồ thị có 2 ô: train loss, và val RMSE với đường chấm của Ridge và moving_avg_6 (val) để biết mô hình đã vượt mốc chưa. Nét đứt = đang chạy, nét liền = đã xong, chấm = epoch tốt nhất.

Cách đọc: val RMSE giảm rồi đi ngang là bình thường; train loss giảm mà val RMSE tăng là overfit (early stopping sẽ dừng); val RMSE nhảy loạn là `lr` quá cao.

## 4. How to do
1. Chạy xong `preprocess` (05) và `baselines` (06).
2. **Chạy thử nhanh:** `python -m src.train --model all --seeds 0 --set train.epochs=2` — chỉ để kiểm tra pipeline, kết quả không có ý nghĩa.
3. **Chạy thật:** `python -m src.train --model all` (3 seed × 2 mô hình).
4. **Mở terminal thứ hai và chạy `python -m src.monitor --watch 10`** để xem training trực tiếp. Sau khi xong, xem `outputs/models/*/seed*/history.csv` (hoặc `outputs/figures/training_curves.png` sau khi chạy `evaluate`) để kiểm tra T2–T5.
5. Nếu val RMSE dao động mạnh: giảm `lr`, tăng `weight_decay`/`dropout`, hoặc thử `train.loss=huber`; ghi thay đổi vào `11_decisions_log.md`.
6. Việc chọn siêu tham số chỉ dựa trên **val**. Sau khi đã chốt, mới chạy `evaluate` để nhìn test.
