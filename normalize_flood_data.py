import pandas as pd

# Đọc file CSV
df = pd.read_csv('flood_points_3k.csv')

# Lấy danh sách các cột cần chuẩn hóa (tất cả trừ cột 'flood')
features_to_normalize = [col for col in df.columns if col != 'flood']

# Chuẩn hóa Min-Max cho từng feature
for feature in features_to_normalize:
    x_min = df[feature].min()
    x_max = df[feature].max()
    
    # Tránh chia cho 0 nếu tất cả giá trị giống nhau
    if x_max - x_min != 0:
        df[feature] = (df[feature] - x_min) / (x_max - x_min)
    else:
        df[feature] = 0

# In ra một vài dòng đầu để kiểm tra
print("Dữ liệu sau khi chuẩn hóa:")
print(df.head())

# Lưu file đã chuẩn hóa (tùy chọn)
df.to_csv('flood_points_3k_normalized.csv', index=False)
print("\nĐã lưu file đã chuẩn hóa vào 'flood_points_3k_normalized.csv'")
