# 11 — Decisions log: quyết định đã chốt và quyết định mở

## 1. Purpose
Ghi lại mọi lựa chọn thiết kế cùng lý do, và đánh dấu những chỗ cần **thảo luận với bạn** trước khi chạy thật.

## 2. IO
- **Input:** kết luận từ EDA (`outputs/eda/eda_summary.md`), kết quả val, trao đổi giữa hai bên.
- **Output:** file này được cập nhật (thêm dòng mới, không xoá lịch sử); thay đổi tương ứng trong `configs/config.yaml`.

### Đã chốt
| ID | Quyết định | Lý do | Cấu hình |
|---|---|---|---|
| D1 | Framework **PyTorch** | Lấy attention weights dễ | — |
| D2 | Dự đoán **t+6 (1 giờ)** từ window đến t | Bạn chọn; khó hơn t+1 và có ý nghĩa thực tế hơn | `window.horizon: 6` |
| D5 | Chia **70/15/15 theo thời gian**, không shuffle | Chuỗi thời gian; shuffle gây rò rỉ | `split.*` |
| D6 | `lookback = 36` (6 giờ) | Điểm khởi đầu hợp lý; **xác nhận lại bằng ACF** ở EDA | `window.lookback` |
| D7 | Giữ feature giờ/thứ dạng sin/cos | Có chu kỳ ngày (kiểm ở EDA) | `data.add_time_features` |
| D8 | 3 seed, báo cáo mean ± std | Giảm nhiễu do khởi tạo | `train.seeds` |
| D9 | Dùng lịch sử `Appliances` làm input | Chuẩn cho dự báo tự hồi quy | `data.use_target_as_input` |
| D10 | Bỏ `rv1`, `rv2` | Hai biến ngẫu nhiên của dataset gốc | `data.drop_cols` |
| D11 | Dataset do bạn tự tải vào `data/raw/` | Môi trường của mình không truy cập UCI | — |
| D12 | Baseline: persistence, moving average, daily naive, Ridge | Mốc đơn giản để chứng minh DL có giá trị | `06` |
| D13 | Cỡ mô hình tương đương (~6–7 vạn tham số) | So sánh công bằng | `07` |
| D14 | Token CLS + Pre-LN cho Transformer | Attention từ CLS dễ diễn giải | `transformer.pooling: cls` |
| D15 | Early stopping theo **val RMSE bằng Wh** | Metric đúng đơn vị báo cáo | `train.patience` |

### Đã chốt sau EDA (2026-11-25)
| ID | Quyết định | Bằng chứng (EDA thật) | Cấu hình |
|---|---|---|---|
| D3 | Dùng **`log1p`** cho target (theo chọn của bạn) | skew 3.39 → 1.19 sau log1p | `data.target_transform: log1p` |
| D4 | Loss **MSE** là chính; Huber chỉ là thí nghiệm đối chứng | 10,8% điểm là đỉnh thật, RMSE đo bằng Wh | `train.loss: mse` |
| D16 | `lookback = 36` trước; sau đó thử `144` làm thí nghiệm | ACF: lag6=0,32; lag36=0,02; lag144=0,22 | `window.lookback: 36` |
| D18 | Drift train→test (tối đa 3,14 std) chỉ **nêu rõ trong báo cáo** | `split_drift.png`, `split_drift_zscore.csv` | — |

Lưu ý rủi ro của D3: khi quy ngược `expm1` về Wh, dự đoán có xu hướng thấp hơn trung bình (gần trung vị) nên RMSE bằng Wh có thể xấu hơn. Nếu val RMSE kém, chạy thêm `--set data.target_transform=none` để so sánh và ghi kết quả vào đây.

### Đã chốt sau kết quả chạy đầu (2026-11-26)
| ID | Quyết định | Lý do |
|---|---|---|
| D20 | **Không** thêm baseline `train_mean`; thay vào đó nâng V3 (vượt Ridge) thành **bắt buộc** và thêm kiểm tra ý nghĩa thống kê V3b | `moving_avg_all` (R² ≈ 0,02) đã là mốc gần hằng số; Ridge (RMSE 81,72) mới là mốc có ý nghĩa. Kết quả đầu: Transformer 82,39 ± 2,56 ≈ Ridge, nên V3 báo FAIL ở điểm ước lượng |

### Đã chốt: ablation (2026-11-26)
| ID | Quyết định | Cách làm |
|---|---|---|
| D21 | **Làm** ablation bỏ nhóm nhiệt độ (`T1`–`T9`, `T_out`, `Tdewpoint`) để kiểm tra ảnh hưởng của drift | Chạy vào thư mục riêng (`outputs_no_temp`), so sánh bằng `compare_ablation`, **chọn theo val**, nêu test như quan sát. Chi tiết `doc/14`. Kết quả: *chưa chạy, điền sau* |

### Quyết định MỞ còn lại
| ID | Câu hỏi | Lựa chọn | Thiên hướng của mình | Cách quyết |
|---|---|---|---|---|
| D17 | Có thêm multi-horizon (t+1, t+6, t+36)? | không / có | **Không** trong phiên bản đầu, tránh phình scope | Bàn nếu cần mở rộng |
| D19 | Kết luận Transformer vs LSTM nếu CI chứa 0 | “không khác biệt” | Đồng ý ghi đúng như vậy | — |

### Thí nghiệm đã quyết định nhưng CHƯA làm
| ID | Thí nghiệm | Mục đích | Cách chạy | Trạng thái |
|---|---|---|---|---|
| D21 | Ablation bỏ nhóm nhiệt độ | Kiểm tra ảnh hưởng của drift nhiệt độ | `scripts\run_ablation_no_temp.ps1` (xem `14`) | Chưa chạy trên dữ liệu thật |
| D3-b | So `log1p` với `none` trên val | MAE/MAPE gợi ý log1p làm dự đoán thấp ở mức thấp/đỉnh | `--set data.target_transform=none` + thư mục riêng | Chưa chạy |
| D16-b | `lookback = 144` | Xem attention có chú ý tới lag 1 ngày không | `--set window.lookback=144` + thư mục riêng | Chưa chạy |
| D4-b | Huber làm đối chứng | Kiểm tra độ ổn định | `--set train.loss=huber` + thư mục riêng | Chưa chạy |

## 3. Success Criteria
- Mọi thay đổi `config.yaml` có một dòng tương ứng ở đây (ID, ngày, lý do).
- Không còn quyết định mở nào chưa có người chốt **trước khi chạy `train` bản cuối**.
- Quyết định dựa trên dữ liệu phải trỏ được tới file bằng chứng (vd. `outputs/eda/...`, val RMSE).

## 4. How to do
1. Sau khi có `outputs/eda/eda_summary.md`, đọc mục “Phát hiện & gợi ý”.
2. Với mỗi quyết định mở (D17, D19) và mỗi thí nghiệm chưa làm, trao đổi với nhau, chọn, rồi điền cột “Cấu hình” + ngày + lý do.
3. Sửa `configs/config.yaml` tương ứng; chạy lại từ `preprocess` nếu đổi `data.*`, `window.*`, `split.*`.
4. Quy tắc chọn: **chỉ dùng val để chọn**, test chỉ để báo cáo cuối.
5. Thêm quyết định mới ở cuối bảng với ID tiếp theo (D22, D23, …).
