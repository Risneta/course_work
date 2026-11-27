# 03 — Data setup: tải và kiểm tra dataset

## 1. Purpose
Đưa dataset gốc vào đúng chỗ và xác nhận nó nguyên vẹn trước khi làm bất cứ thứ gì khác.

## 2. IO
- **Input:** file `energydata_complete.csv` tải từ https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction (bạn tải tay — môi trường của mình không truy cập được UCI).
- **Output:** file đặt tại `data/raw/energydata_complete.csv`; báo cáo kiểm tra nằm trong `outputs/eda/eda_report.json` (mục `validation`) và `data/processed/preprocess_report.json`.

## 3. Success Criteria
Kiểm tra bởi `validate_raw()` trong `src/data.py`:
| Kiểm tra | Mức | Kỳ vọng |
|---|---|---|
| Có cột `date` và `Appliances` | **Lỗi, dừng** | có |
| Mọi cột (trừ `date`) là số | **Lỗi, dừng** | có |
| Số dòng | Cảnh báo | 19 735 |
| Ô thiếu | Cảnh báo (sẽ nội suy tuyến tính) | 0 |
| Timestamp trùng | Cảnh báo | 0 |
| Bước thời gian ≠ 10 phút | Cảnh báo | 0 |

Nếu cảnh báo xuất hiện, dừng lại và bàn với nhau trước khi tiếp tục (xem `11_decisions_log.md`).

## 4. How to do
1. Mở trang UCI ở trên → tải dataset (zip) → giải nén.
2. Copy `energydata_complete.csv` vào `data/raw/`.
3. Chạy `python -m src.eda`; đọc phần đầu của `outputs/eda/eda_summary.md` (số dòng, giai đoạn, thiếu, trùng, bước không đều).
4. Nếu số dòng/cột khác kỳ vọng: kiểm tra đã lấy đúng file `energydata_complete.csv` (không phải file train/test tách sẵn của bản khác).
5. Chưa có dữ liệu thật mà muốn thử pipeline: `python scripts/make_synthetic.py`, rồi thêm `--data-path data/synthetic/energydata_synthetic.csv` vào các lệnh. Kết quả như vậy **chỉ để kiểm tra code**.
