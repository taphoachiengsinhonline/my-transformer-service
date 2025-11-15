# File: export_data.py
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# --- THAY ĐỔI THÔNG TIN KẾT NỐI CỦA BẠN VÀO ĐÂY ---
MONGO_URI = "mongodb+srv://maytinhthaikhang_db_user:Dat15122004@kq.cz4y4z8.mongodb.net/?appName=kq"
DB_NAME = "test" # Tên database của bạn
# ----------------------------------------------------

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db.results # Giả sử collection của bạn tên là 'results'

# Lấy toàn bộ dữ liệu
cursor = collection.find({})
results_list = list(cursor)

if not results_list:
    print("Không tìm thấy dữ liệu!")
else:
    # Chuyển thành DataFrame của Pandas để dễ xử lý
    df = pd.DataFrame(results_list)

    # Chuyển cột 'ngay' thành kiểu datetime để sắp xếp
    df['date_obj'] = pd.to_datetime(df['ngay'], format='%d/%m/%Y')
    df = df.sort_values(by='date_obj').reset_index(drop=True)

    # Lưu ra file CSV
    df.to_csv('results.csv', index=False)
    print(f"Đã xuất thành công {len(df)} dòng dữ liệu ra file results.csv")

client.close()