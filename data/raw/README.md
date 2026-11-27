# data/raw/

Đặt file **`energydata_complete.csv`** (UCI Appliances Energy Prediction) vào thư mục này.

- Nguồn: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
- Tải file zip/csv từ trang trên, giải nén, lấy `energydata_complete.csv`.
- Không sửa file gốc. Mọi bước xử lý đều ghi ra `data/processed/` và `outputs/`.

Kiểm tra nhanh sau khi đặt file:

```bash
python -m src.eda
```
Nếu số dòng khác 19735 hoặc thiếu cột, EDA sẽ in cảnh báo (xem `doc/03_data_setup.md`).
