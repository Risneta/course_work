# 13 — Notebook báo cáo và biểu đồ toàn bộ quá trình huấn luyện

## 1. Purpose
Có một chỗ duy nhất để **xem và trình bày** toàn bộ bài làm: bài toán, EDA, tiền xử lý, baseline, **mọi run huấn luyện (mọi model, mọi seed) trên biểu đồ, lý do dừng sớm**, đánh giá, attention, kết luận. Notebook **không huấn luyện lại**, chỉ đọc kết quả đã có.

## 2. IO
- **Input:** `outputs/` (EDA, baselines, models, metrics, figures, attention), `data/processed/`, `configs/config.yaml`. Cần `history.csv` và `config_snapshot.json` trong từng `outputs/models/{model}/seed{S}/`.
- **Output:**
  | File | Sinh bởi | Nội dung |
  |---|---|---|
  | `notebooks/report.ipynb` | (đã viết sẵn) | Notebook trình bày, chạy bằng *Run All* |
  | `outputs/figures/training_overview.png` | `python -m src.training_report` | Mọi run trên cùng một hình: train loss, val RMSE (có epoch tốt nhất và điểm dừng), learning rate |
  | `outputs/figures/training_per_run.png` | `python -m src.training_report` | Lưới từng run: train loss + val RMSE, vùng chờ early stopping |
  | `outputs/models/training_runs_table.csv` | `python -m src.training_report` | Epoch chạy, epoch tốt nhất, cải thiện so với epoch 1, số lần giảm lr, thời gian, **lý do dừng**, nhận xét |
- Mục lục notebook: 1 Bài toán · 2 EDA · 3 Tiền xử lý · 4 Baseline · 5 Huấn luyện (đường cong mọi run, lý do dừng) · 6 Đánh giá · 7 Attention · 7b Ablation nhiệt độ · 8 Kết luận · 9 Câu hỏi hay gặp khi bảo vệ.
- Code liên quan: `src/training_report.py` (không cần torch), `src/monitor.py` (xem trực tiếp lúc đang train).

### Lý do dừng được xác định thế nào
`training_report.py` tái hiện đúng logic của `train.py` trên `history.csv`: một epoch là “cải thiện” nếu val RMSE thấp hơn mức tốt nhất trước đó hơn 1e-6; dừng khi số epoch liên tiếp không cải thiện đạt `patience` (đọc từ `config_snapshot.json` của chính run đó). Kết quả là một trong ba lý do: **Early stopping** (kèm epoch tốt nhất và epoch dừng), **Hết số epoch tối đa**, hoặc **Chưa kết thúc / bị ngắt**.
Cột `nhận_xét` so val RMSE *sau* epoch tốt nhất với *trước* đó (không so với chính điểm tốt nhất, vì điểm tốt nhất luôn thấp hơn trung bình của một chuỗi nhiễu): tăng > 1 Wh trong khi train loss vẫn giảm → “nghi overfit”; ngược lại → “đi ngang / đã bão hoà”.

## 3. Success Criteria
| # | Kiểm tra | Kỳ vọng |
|---|---|---|
| N1 | Notebook chạy *Run All* không lỗi | đúng; file thiếu chỉ in “(chưa có …)” |
| N2 | Mục 5 hiện đủ mọi (model, seed) đã train | số cột trong `training_per_run.png` bằng số seed |
| N3 | Mỗi run có đúng một lý do dừng | cột `lý_do_dừng` không rỗng |
| N4 | `epoch_tốt_nhất` trùng `best_epoch` trong `train_summary.json` | khớp |
| N5 | Số trong mục 8 (kết luận) khớp `outputs/results_report.md` | khớp |
| N6 | Phần Hạn chế được đọc lại và sửa nếu chạy lại với kết quả khác | đã rà |

Đã kiểm thử bằng dữ liệu **giả lập** (đầy đủ và trống hoàn toàn); chưa chạy trên kết quả thật của bạn.

## 4. How to do
1. Hoàn tất pipeline: `train` → `evaluate` → `attention`.
2. Chạy `python -m src.training_report` (in bảng lý do dừng và lưu 2 hình + CSV).
3. Cài Jupyter nếu chưa có: `python -m pip install notebook ipykernel` (hoặc mở file trong VS Code, chọn kernel Python của môi trường).
4. Mở `notebooks/report.ipynb` → *Run All*.
5. Với mục 5.3 và 5.4, đối chiếu từng run: epoch tốt nhất, lý do dừng, nhận xét; kiểm tra N4.
6. Đọc lại mục 8 (kết luận, hạn chế) và sửa chữ nếu kết quả của bạn khác với câu mẫu.
7. Để nộp: *File → Save and Export Notebook As → HTML* (hoặc PDF) sau khi đã Run All.
