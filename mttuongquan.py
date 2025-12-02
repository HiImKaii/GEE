import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
file_path = r"D:\25-26_HKI_DATN_QuanVX\train\data\flood_data_with_13_features.csv"
df = pd.read_csv(file_path)

# Loại bỏ cột nhãn (flood)
df_features = df.drop(columns=['flood'])

# Định nghĩa tên cột chuẩn (tiếng Việt có dấu hoặc viết tắt ngắn gọn)
column_mapping = {
    'lulc': 'LULC',
    'Density_River': 'Mật độ sông',
    'Density_Road': 'Mật độ đường',
    'Distan2river': 'Khoảng cách sông',
    'Distan2road_met': 'Khoảng cách đường',
    'aspect': 'Hướng sườn',
    'curvature': 'Độ cong',
    'dem': 'DEM',
    'flowDir': 'Hướng dòng chảy',
    'slope': 'Độ dốc',
    'twi': 'TWI',
    'NDVI': 'NDVI',
    'rainfall': 'Lượng mưa'
}

# Đổi tên cột
df_features = df_features.rename(columns=column_mapping)

# Tính ma trận tương quan
corr_matrix = df_features.corr()

# Tạo mask cho nửa trên (chỉ hiển thị nửa dưới)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Thiết lập kích thước figure
plt.figure(figsize=(14, 12))

# Vẽ ma trận tương quan với nửa dưới
sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True,  # Hiển thị giá trị
    fmt='.2f',   # Định dạng 2 chữ số thập phân
    cmap='coolwarm',  # Bảng màu
    center=0,    # Tâm màu tại 0
    square=True, # Ô vuông
    linewidths=0.5,  # Độ rộng đường kẻ
    cbar_kws={"shrink": 0.8},  # Kích thước thanh màu
    vmin=-1,     # Giá trị min
    vmax=1       # Giá trị max
)

# Thiết lập tiêu đề và labels
plt.title('Ma trận tương quan các chỉ số', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')

# Xoay labels trục x
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Căn chỉnh layout
plt.tight_layout()

# Lưu hình
output_path = r"D:\25-26_HKI_DATN_QuanVX\train\data\correlation_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Đã lưu ma trận tương quan tại: {output_path}")

# Hiển thị
plt.show()

# In thông tin thống kê về tương quan
print("\n=== Thống kê tương quan ===")
print(f"Số lượng chỉ số: {len(df_features.columns)}")
print(f"\nCác cặp có tương quan cao nhất (|r| > 0.7):")

# Tìm các cặp có tương quan cao
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Chỉ số 1': corr_matrix.columns[i],
                'Chỉ số 2': corr_matrix.columns[j],
                'Hệ số tương quan': corr_matrix.iloc[i, j]
            })

if high_corr:
    high_corr_df = pd.DataFrame(high_corr)
    high_corr_df = high_corr_df.sort_values('Hệ số tương quan', key=abs, ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("Không có cặp nào có tương quan > 0.7")
