# 14 — Ablation D21: bỏ nhóm nhiệt độ để kiểm tra ảnh hưởng của drift

## 1. Purpose
EDA cho thấy test rơi vào giai đoạn ấm hơn train: nhiệt độ trong nhà (`T1`–`T9`) lệch +1,9 đến +3,1 độ lệch chuẩn, `T_out` lệch +2,1, `Tdewpoint` lệch +1,8; còn target `Appliances` gần như không lệch (−0,02). Nhiệt độ lại chỉ tương quan yếu với target (|corr| cao nhất ~0,12). Câu hỏi: **mô hình có đang bị thiệt vì phải ngoại suy trên nhiệt độ không?** Cách trả lời: huấn luyện lại không dùng nhóm nhiệt độ và so sánh với bản đầy đủ.

Nhóm bị bỏ khỏi input: `T1`–`T9`, `T_out`, `Tdewpoint` (11 cột). Giữ lại: lịch sử `Appliances`, `lights`, toàn bộ độ ẩm (`RH_*`), `Press_mm_hg`, `Windspeed`, `Visibility`, feature giờ/thứ. Target và cách tạo nhãn **không đổi**.

## 2. IO
- **Input:** `data/raw/energydata_complete.csv`, `configs/config.yaml`; kết quả bản đầy đủ đã có trong `outputs/` (cần `evaluate` đã chạy: `predictions/`, `metrics/per_seed_metrics.csv`, `baselines/`).
- **Cấu hình mới:** `data.exclude_features: [...]` (mặc định rỗng). Tên không tồn tại hoặc chứa target sẽ báo lỗi.
- **Output** — toàn bộ ghi vào thư mục **riêng**, không đụng kết quả chính:
  | Vị trí | Nội dung |
  |---|---|
  | `data/processed_no_temp/` | npz đã scale của bản bỏ nhiệt độ (F = 20 thay vì 31) |
  | `outputs_no_temp/` | baselines, models, predictions, metrics, figures của bản ablation |
  | `outputs_no_temp/ablation_comparison.md` | **bảng so sánh + kiểm định + kết luận tự động** |
  | `outputs_no_temp/ablation_comparison.csv`, `ablation_comparison.png` | số liệu và hình |
- **Script:** `scripts/run_ablation_no_temp.ps1` (Windows), `scripts/run_ablation_no_temp.sh`. **Code:** `src/data.py` (exclude_features), `src/compare_ablation.py`.

## 3. Success Criteria
| # | Kiểm tra | Kỳ vọng |
|---|---|---|
| X1 | Số feature giảm đúng 11 | `data/processed_no_temp/feature_names.json` có 20 phần tử, không có `T1..T9`, `T_out`, `Tdewpoint` |
| X2 | Nhãn và cửa sổ test **giống hệt** bản đầy đủ | `compare_ablation` dừng nếu `y_raw` khác nhau |
| X3 | Kết quả bản đầy đủ không bị ghi đè | `outputs/` không đổi sau khi chạy ablation |
| X4 | Cùng seed, cùng siêu tham số, cùng tiêu chí dừng ở hai bản | `config_snapshot.json` chỉ khác `exclude_features` và `paths` |
| X5 | **Chọn theo val, không theo test** | bảng có cột val RMSE; kết luận cấu hình chính dựa trên val |

**Cách đọc kết quả (Δ = RMSE bỏ nhiệt độ − RMSE đầy đủ):**
- Δtest < 0 và CI95 không chứa 0, **và** Δval cũng < 0: bỏ nhiệt độ thực sự tốt hơn; cân nhắc dùng bản này làm cấu hình chính.
- Δtest < 0 nhưng Δval ≥ 0 (val và test ngược hướng): dấu hiệu drift. Cấu hình chính vẫn chọn theo val; nêu kết quả test như một quan sát trong báo cáo.
- CI95 chứa 0: chưa có bằng chứng khác biệt; ghi đúng như vậy.
- Ridge thường nhạy với drift nhất (hàm tuyến tính ngoại suy thẳng). Nếu chỉ Ridge cải thiện mà Transformer không, đó cũng là một phát hiện.

**Giới hạn:** bỏ nhiệt độ cũng bỏ luôn tín hiệu mùa, nên không tách riêng được “ngoại suy do drift” khỏi “mất thông tin có ích”. Chỉ một giai đoạn test và 3 seed.

## 4. How to do
1. Bản đầy đủ đã chạy xong `train` và `evaluate` (đã có).
2. Chạy thử nhanh (vài phút, kết quả không có ý nghĩa): `powershell -ExecutionPolicy Bypass -File scripts\run_ablation_no_temp.ps1 -Quick`. Linux/macOS: `QUICK=1 bash scripts/run_ablation_no_temp.sh`.
3. Chạy thật (thời gian xấp xỉ một lần train đầy đủ): `powershell -ExecutionPolicy Bypass -File scripts\run_ablation_no_temp.ps1`.
4. Trong lúc train, mở terminal thứ hai: `python -m src.monitor --watch 10 --set paths.output_dir=outputs_no_temp`.
5. Mở `outputs_no_temp/ablation_comparison.md`; đối chiếu X1–X5.
6. Ghi kết luận vào `doc/11_decisions_log.md` (D21) và vào mục 7b của notebook (chạy lại *Run All*).
7. Muốn thử loại nhóm khác: `--set "data.exclude_features=[RH_1,RH_2]"` với `--set paths.processed_dir=...` và `--set paths.output_dir=...` trỏ sang thư mục riêng.
