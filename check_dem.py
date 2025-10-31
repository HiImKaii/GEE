import rasterio
import numpy as np

file_path = r"D:\prj\feature\gialai_curvature.tif"

with rasterio.open(file_path) as ds:
    data = ds.read(1)
    
    print(f"{'='*60}")
    print(f"PHÂN TÍCH DỮ LIỆU CURVATURE - ĐỘ CONG ĐỊA HÌNH")
    print(f"{'='*60}")
    
    # Thông tin cơ bản
    print(f"\nThông tin file:")
    print(f"  Data type: {data.dtype}")
    print(f"  Data shape: {data.shape[0]} x {data.shape[1]} = {data.size:,} pixels")
    print(f"  CRS: {ds.crs}")
    
    # Xử lý NoData và NaN
    valid_mask = ~np.isnan(data)
    if ds.nodata is not None:
        valid_mask = valid_mask & (data != ds.nodata)
    
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        print("\n⚠ CẢNH BÁO: Không có dữ liệu hợp lệ trong file!")
        exit()
    
    # Thống kê dữ liệu hợp lệ
    print(f"\nDữ liệu hợp lệ:")
    print(f"  Số pixel hợp lệ: {len(valid_data):,} ({len(valid_data)/data.size*100:.2f}%)")
    print(f"  Số pixel không hợp lệ: {data.size - len(valid_data):,} ({(data.size - len(valid_data))/data.size*100:.2f}%)")
    
    # Dải giá trị
    print(f"\n{'='*60}")
    print(f"DẢI GIÁ TRỊ CURVATURE")
    print(f"{'='*60}")
    print(f"  Min:    {valid_data.min():.6f}")
    print(f"  Max:    {valid_data.max():.6f}")
    print(f"  Mean:   {valid_data.mean():.6f}")
    print(f"  Median: {np.median(valid_data):.6f}")
    print(f"  Std:    {np.std(valid_data):.6f}")
    
    # Percentiles
    print(f"\nPhân vị (Percentiles):")
    print(f"  P25:  {np.percentile(valid_data, 25):.6f}")
    print(f"  P50:  {np.percentile(valid_data, 50):.6f}")
    print(f"  P75:  {np.percentile(valid_data, 75):.6f}")
    print(f"  P95:  {np.percentile(valid_data, 95):.6f}")
    
    print(f"\n{'='*60}")

