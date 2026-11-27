# 07 — Models: Transformer Encoder và LSTM

## 1. Purpose
Định nghĩa hai mô hình hồi quy có **cùng giao diện** `(B, L, F) → (B,)` và cỡ tham số tương đương, để so sánh công bằng.

## 2. IO
- **Input:** tensor `(B, L, F)` float32 (đã scale). **Output:** tensor `(B,)` — target đã scale (quy về Wh bằng `inverse_target`).
- **Code:** `src/models.py`. **Tham số:** mục `transformer`, `lstm` trong `configs/config.yaml`.

### TransformerRegressor
```
x (B,L,F) → Linear(F→d_model) → + sinusoidal positional encoding
          → [CLS] token nối vào đầu chuỗi   (pooling = cls)
          → N × EncoderLayer (Pre-LN: LayerNorm → MultiHeadAttention → residual; LayerNorm → FFN(GELU) → residual)
          → LayerNorm → lấy vector CLS → MLP(d→d→1)
```
Mặc định: `d_model=64`, `nhead=4`, `num_layers=2`, `dim_ff=128`, `dropout=0.1`. Có thể đổi `pooling` thành `mean` hoặc `last`.
Mỗi `EncoderLayer` có thể lưu attention `(B, heads, S, S)` qua `forward_with_attention()` (S = L+1 khi có CLS).

### LSTMRegressor
```
x (B,L,F) → LSTM(hidden=64, 2 layers, dropout=0.1) → hidden state cuối của layer trên cùng → MLP(h→h→1)
```
Cùng đầu ra MLP như Transformer để chênh lệch chỉ đến từ phần mã hoá chuỗi.

## 3. Success Criteria
Kiểm tra bằng `tests/test_models.py` (cần torch):
| # | Kiểm tra | Kỳ vọng |
|---|---|---|
| M1 | Output shape | `(B,)` cho cả hai mô hình |
| M2 | Backward chạy được | loss hữu hạn, gradient có |
| M3 | Attention shape | `(B, heads, L+1, L+1)` với `pooling=cls` |
| M4 | Mỗi hàng attention cộng = 1 | sai số `< 1e-4` |
| M5 | `forward_with_attention` cho cùng dự đoán với `forward` | sai khác `< 1e-5`, và trạng thái lưu attention được dọn sau khi gọi |
| M6 | Số tham số cùng bậc | Transformer ≈ LSTM (khoảng 6–7 vạn tham số với cấu hình mặc định; số chính xác in ở `train_summary.json`) |

## 4. How to do
1. Cài torch (`pip install -r requirements.txt`) và chạy `python -m pytest -q tests/test_models.py`.
2. Đổi kiến trúc: sửa mục `transformer`/`lstm` trong config (không sửa code). Khi tăng cỡ một mô hình, chỉnh mô hình kia để số tham số cùng bậc rồi ghi lại vào `11_decisions_log.md`.
3. Muốn thử `pooling` khác cho Transformer: `--set transformer.pooling=mean`. Lưu ý: khi không có CLS, `attention.py` dùng trung bình mọi query thay vì hàng CLS.
4. Thêm mô hình mới: viết lớp trong `models.py`, thêm vào `build_model`, thêm mục config tương ứng.
