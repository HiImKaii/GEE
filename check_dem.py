import rasterio
import numpy as np

file_path = r"D:\Vscode\gee\ESA_WorldCover_30m_2021.tif"

with rasterio.open(file_path) as ds:
    data = ds.read(1)
    print(f"Data type: {data.dtype}")
    print(f"Data shape: {data.shape}")
    print(f"Min value: {data.min()}")
    print(f"Max value: {data.max()}")
    print(f"Mean value: {data.mean():.6f}")
    print(f"NoData value: {ds.nodata}")
    print(f"CRS: {ds.crs}")
    print(f"Bounds: {ds.bounds}")
    
    # Kiểm tra các giá trị unique (đặc biệt quan trọng cho dữ liệu phân loại)
    unique_vals = np.unique(data)
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH DỮ LIỆU ESA WORLDCOVER 2021 (PHÂN LOẠI)")
    print(f"{'='*60}")
    print(f"Số lớp phân loại (unique values): {len(unique_vals)}")
    print(f"Các lớp có trong dữ liệu: {unique_vals}")
    
    # Kiểm tra có bao nhiêu pixel là NoData
    if ds.nodata is not None:
        nodata_count = np.sum(data == ds.nodata)
        print(f"\nSố pixel NoData: {nodata_count} ({nodata_count/data.size*100:.2f}%)")
    
    # Thống kê vùng không phải NoData
    if ds.nodata is not None:
        valid_data = data[data != ds.nodata]
    else:
        valid_data = data
    
    # Đếm số lượng pixel cho mỗi lớp ESA WorldCover
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ SỐ LƯỢNG PIXEL CHO MỖI LỚP ESA WORLDCOVER")
    print(f"{'='*60}")
    
    # ESA WorldCover legend
    lulc_labels = {
        10: 'Tree cover',
        20: 'Shrubland',
        30: 'Grassland',
        40: 'Cropland',
        50: 'Built-up',
        60: 'Bare / sparse vegetation',
        70: 'Snow and ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        95: 'Mangroves',
        100: 'Moss and lichen'
    }
    
    total_valid_pixels = len(valid_data)
    for val in sorted(np.unique(valid_data)):
        count = np.sum(valid_data == val)
        percentage = (count / total_valid_pixels) * 100
        label = lulc_labels.get(int(val), f'Không xác định (Giá trị {val})')
        print(f"Lớp {int(val):2d} - {label:20s}: {count:12,} pixels ({percentage:6.2f}%)")
    
    print(f"\nTổng pixel hợp lệ: {total_valid_pixels:,}")
    
    # Kiểm tra xem có giá trị nào ngoài phạm vi ESA WorldCover không
    print(f"\n{'='*60}")
    print(f"KIỂM TRA TÍNH HỢP LỆ")
    print(f"{'='*60}")
    valid_esa_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    unexpected_values = valid_data[~np.isin(valid_data, valid_esa_values)]
    if len(unexpected_values) > 0:
        print(f"⚠ CẢNH BÁO: Có {len(unexpected_values)} pixel với giá trị ngoài phạm vi ESA WorldCover!")
        print(f"Các giá trị bất thường: {np.unique(unexpected_values)}")
    else:
        print(f"✓ Tất cả giá trị đều nằm trong phạm vi hợp lệ ESA WorldCover")
    
    # Thống kê cơ bản
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ CƠ BẢN")
    print(f"{'='*60}")
    print(f"Min hợp lệ: {valid_data.min():.0f}")
    print(f"Max hợp lệ: {valid_data.max():.0f}")
    print(f"Mean hợp lệ: {valid_data.mean():.2f}")
    print(f"Median: {np.median(valid_data):.0f}")
    
    # Mode cho dữ liệu ESA WorldCover
    unique, counts = np.unique(valid_data, return_counts=True)
    mode_value = unique[np.argmax(counts)]
    mode_label = lulc_labels.get(int(mode_value), f'Giá trị {int(mode_value)}')
    print(f"Mode (giá trị phổ biến nhất): {int(mode_value)} ({mode_label})")

