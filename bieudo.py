import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Đọc dữ liệu từ file CSV
csv_file = r"D:\prj\results\dien_tich.csv"
df = pd.read_csv(csv_file, encoding='utf-8')

# Tạo tên ngắn gọn cho các mô hình
model_names = []
for name in df['Tên ảnh']:
    # Tách lấy tên mô hình từ đường dẫn
    parts = name.split('/')
    model_type = parts[0].upper()  # rf, svm, xgb
    # Đổi SVR thành SVM
    if model_type == 'SVR':
        model_type = 'SVM'
    optimization = parts[1].split('_')[-2].upper()  # pso, puma, rs
    model_names.append(f"{model_type}-{optimization}")

# Danh sách tên cột ngưỡng
thresholds = ['Ngưỡng 1 (km²)', 'Ngưỡng 2 (km²)', 'Ngưỡng 3 (km²)', 
              'Ngưỡng 4 (km²)', 'Ngưỡng 5 (km²)']

# Định nghĩa màu sắc cho từng mô hình (9 màu khác nhau)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Định nghĩa pattern cho các ngưỡng
patterns = ['', '...', 'xxx', '///', '\\\\\\']  # Sử dụng pattern có sẵn
pattern_labels = ['Ngưỡng 1', 'Ngưỡng 2', 'Ngưỡng 3', 'Ngưỡng 4', 'Ngưỡng 5']

# Tạo figure với 1 biểu đồ duy nhất
fig, ax = plt.subplots(figsize=(16, 10))

# Đặt nền trắng đơn giản
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Vẽ biểu đồ cột xếp chồng
x_pos = np.arange(len(model_names))
bar_width = 0.2  # Độ rộng cột

# Tạo mảng để lưu bottom position cho từng cột xếp chồng
bottom = np.zeros(len(model_names))

# Vẽ cột cho từng ngưỡng, xếp chồng lên nhau
for idx, threshold in enumerate(thresholds):
    # Lấy dữ liệu cho ngưỡng hiện tại
    data = df[threshold].values
    
    # Vẽ biểu đồ cột xếp chồng với pattern
    bars = ax.bar(x_pos, data, bar_width, bottom=bottom, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8, hatch=patterns[idx])
    
    # Cập nhật bottom position cho lần lặp tiếp theo
    bottom += data


# Thiết lập nhãn và tiêu đề
ax.set_xlabel('Mô hình', fontsize=12, fontweight='bold')
ax.set_ylabel('Diện tích (km²)', fontsize=12, fontweight='bold')

# Thiết lập trục x
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)

# Giới hạn trục y để giảm chiều cao cột
max_value = bottom.max()
ax.set_ylim(0, max_value * 1.15)  # Thêm 15% không gian phía trên

# Thêm lưới ngang để dễ đọc
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Định dạng trục y
ax.ticklabel_format(style='plain', axis='y')

# Tạo legend cho màu sắc (mô hình)
legend_patches_models = []
for model_name, color in zip(model_names, colors):
    patch = mpatches.Patch(facecolor=color, label=model_name, edgecolor='black', linewidth=0.5)
    legend_patches_models.append(patch)

# Tạo legend cho pattern (ngưỡng)
legend_patches_patterns = []

# Ngưỡng 1: không pattern
patch1 = mpatches.Patch(facecolor='gray', edgecolor='black', 
                        label='Ngưỡng 1', linewidth=0.8)
legend_patches_patterns.append(patch1)

# Ngưỡng 2: chấm nhỏ
patch2 = mpatches.Patch(facecolor='gray', edgecolor='black', 
                        hatch='...', label='Ngưỡng 2', linewidth=0.8)
legend_patches_patterns.append(patch2)

# Ngưỡng 3: chữ x
patch3 = mpatches.Patch(facecolor='gray', edgecolor='black', 
                        hatch='xxx', label='Ngưỡng 3', linewidth=0.8)
legend_patches_patterns.append(patch3)

# Ngưỡng 4: gạch chéo phải
patch4 = mpatches.Patch(facecolor='gray', edgecolor='black', 
                        hatch='///', label='Ngưỡng 4', linewidth=0.8)
legend_patches_patterns.append(patch4)

# Ngưỡng 5: gạch chéo trái
patch5 = mpatches.Patch(facecolor='gray', edgecolor='black', 
                        hatch='\\\\\\', label='Ngưỡng 5', linewidth=0.8)
legend_patches_patterns.append(patch5)

# Hiển thị 2 legend: 1 cho mô hình, 1 cho ngưỡng
legend1 = ax.legend(handles=legend_patches_models, 
                   loc='upper left',
                   fontsize=9,
                   title='Mô hình',
                   title_fontsize=11,
                   frameon=True,
                   fancybox=False,
                   shadow=False,
                   ncol=3,
                   bbox_to_anchor=(0, 1.02),
                   columnspacing=1.0,
                   handlelength=1.5)

legend2 = ax.legend(handles=legend_patches_patterns, 
                   loc='upper right',
                   fontsize=9,
                   title='Ngưỡng',
                   title_fontsize=11,
                   frameon=True,
                   fancybox=False,
                   shadow=False,
                   bbox_to_anchor=(1, 1.02))

# Thêm lại legend1 vì legend2 đã ghi đè
ax.add_artist(legend1)

# Tùy chỉnh khung chú giải
for legend in [legend1, legend2]:
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('0.8')
    frame.set_linewidth(1.2)
    frame.set_alpha(0.95)

# Điều chỉnh layout
plt.tight_layout(rect=[0, 0, 1, 0.94])

# Lưu hình
output_file = r"D:\prj\results\bieu_do_phan_bo_dien_tich.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ tại: {output_file}")

# Hiển thị biểu đồ
plt.show()
