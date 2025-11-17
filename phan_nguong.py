"""
Script phân ngưỡng các ảnh TIFF có giá trị từ 0-1 thành 5 ngưỡng
Ngưỡng 1: 0-0.2 -> giá trị 1
Ngưỡng 2: 0.2-0.4 -> giá trị 2
Ngưỡng 3: 0.4-0.6 -> giá trị 3
Ngưỡng 4: 0.6-0.8 -> giá trị 4
Ngưỡng 5: 0.8-1.0 -> giá trị 5
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path


def xu_ly_tiff(duong_dan_dau_vao, duong_dan_dau_ra):
    """
    Đọc file TIFF, phân ngưỡng và lưu kết quả
    
    Args:
        duong_dan_dau_vao: Đường dẫn file TIFF đầu vào
        duong_dan_dau_ra: Đường dẫn file TIFF đầu ra
    """
    print(f"Đang xử lý: {duong_dan_dau_vao}")
    
    # Đọc file TIFF
    try:
        with rasterio.open(duong_dan_dau_vao) as src:
            # Đọc dữ liệu
            data = src.read(1)  # Đọc band đầu tiên
            
            # Lưu metadata
            profile = src.profile.copy()
            
            # Phân ngưỡng
            print("Đang phân ngưỡng...")
            data_phan_nguong = np.zeros_like(data, dtype=np.uint8)
            
            # Xử lý các giá trị hợp lệ (không phải NoData)
            no_data_value = src.nodata
            
            if no_data_value is not None:
                mask = data != no_data_value
            else:
                # Nếu không có NoData, coi giá trị âm hoặc >1 là không hợp lệ
                mask = (data >= 0) & (data <= 1) & (~np.isnan(data))
            
            # Phân ngưỡng cho các giá trị hợp lệ
            data_phan_nguong[mask & (data <= 0.2)] = 1
            data_phan_nguong[mask & (data > 0.2) & (data <= 0.4)] = 2
            data_phan_nguong[mask & (data > 0.4) & (data <= 0.6)] = 3
            data_phan_nguong[mask & (data > 0.6) & (data <= 0.8)] = 4
            data_phan_nguong[mask & (data > 0.8)] = 5
            
            # Giữ nguyên NoData
            data_phan_nguong[~mask] = 0
            
            # Cập nhật profile cho output
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=0
            )
            
            # Ghi file output
            with rasterio.open(duong_dan_dau_ra, 'w', **profile) as dst:
                dst.write(data_phan_nguong, 1)
            
            print(f"Đã lưu: {duong_dan_dau_ra}")
            
    except Exception as e:
        print(f"Không thể đọc file: {duong_dan_dau_vao}")
        print(f"Lỗi: {e}")
        return


def xu_ly_thu_muc(thu_muc_goc):
    """
    Xử lý tất cả các file TIFF trong thư mục và các thư mục con
    
    Args:
        thu_muc_goc: Đường dẫn thư mục gốc chứa các file TIFF
    """
    thu_muc_goc = Path(thu_muc_goc)
    
    if not thu_muc_goc.exists():
        print(f"Thư mục không tồn tại: {thu_muc_goc}")
        return
    
    # Tìm tất cả file TIFF
    cac_file_tiff = []
    for root, dirs, files in os.walk(thu_muc_goc):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                # Bỏ qua các file .aux.xml
                if not file.endswith('.aux.xml'):
                    cac_file_tiff.append(Path(root) / file)
    
    print(f"Tìm thấy {len(cac_file_tiff)} file TIFF")
    
    # Xử lý từng file
    for file_tiff in cac_file_tiff:
        # Tính đường dẫn tương đối so với thư mục gốc
        duong_dan_tuong_doi = file_tiff.relative_to(thu_muc_goc)
        
        # Tạo đường dẫn output với thư mục "thresholded"
        thu_muc_output = thu_muc_goc / "thresholded" / duong_dan_tuong_doi.parent
        thu_muc_output.mkdir(parents=True, exist_ok=True)
        
        # Tạo tên file output
        ten_file_output = file_tiff.stem + "_thresholded.tif"
        file_output = thu_muc_output / ten_file_output
        
        # Xử lý file
        try:
            xu_ly_tiff(str(file_tiff), str(file_output))
        except Exception as e:
            print(f"Lỗi khi xử lý {file_tiff}: {e}")
    
    print("\nHoàn thành!")
    print(f"Kết quả được lưu tại: {thu_muc_goc / 'thresholded'}")


if __name__ == "__main__":
    # Đường dẫn thư mục chứa các file TIFF
    thu_muc_du_lieu = r"D:\prj\results\map"
    
    print("="*60)
    print("CHƯƠNG TRÌNH PHÂN NGƯỠNG ẢNH TIFF")
    print("="*60)
    print(f"Thư mục dữ liệu: {thu_muc_du_lieu}")
    print("Phân ngưỡng:")
    print("  - Ngưỡng 1: 0.0 - 0.2  -> Giá trị 1")
    print("  - Ngưỡng 2: 0.2 - 0.4  -> Giá trị 2")
    print("  - Ngưỡng 3: 0.4 - 0.6  -> Giá trị 3")
    print("  - Ngưỡng 4: 0.6 - 0.8  -> Giá trị 4")
    print("  - Ngưỡng 5: 0.8 - 1.0  -> Giá trị 5")
    print("="*60)
    print()
    
    # Xử lý thư mục
    xu_ly_thu_muc(thu_muc_du_lieu)
