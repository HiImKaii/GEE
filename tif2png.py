import rasterio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def tif_to_png(input_tif, output_png):
    """
    Chuyển đổi file TIFF thành PNG với hiển thị màu sắc
    
    Args:
        input_tif: Đường dẫn đến file TIFF đầu vào
        output_png: Đường dẫn đến file PNG đầu ra
    """
    # Đọc file TIFF
    with rasterio.open(input_tif) as src:
        # Đọc dữ liệu
        data = src.read(1)  # Đọc band đầu tiên
        
        # Loại bỏ giá trị NoData (-9999.0)
        nodata_value = -9999.0
        data_masked = np.ma.masked_where(data == nodata_value, data)
        
        # Chuẩn hóa dữ liệu về khoảng 0-255 (chỉ tính với dữ liệu hợp lệ)
        data_min = data_masked.min()
        data_max = data_masked.max()
        
        # Tạo ảnh màu bằng colormap
        plt.figure(figsize=(10, 8))
        plt.imshow(data_masked, cmap='viridis')  # Sử dụng colormap viridis, NoData sẽ không hiển thị
        plt.colorbar(label='Giá trị')
        plt.title('Rainfall 30m')
        plt.axis('off')
        
        # Lưu file PNG với colormap
        plt.savefig(output_png, bbox_inches='tight', dpi=300, pad_inches=0.1)
        plt.close()
        
        print(f"Đã chuyển đổi thành công: {output_png}")
        print(f"Kích thước ảnh: {data.shape}")
        print(f"Giá trị min: {data_min:.2f}, max: {data_max:.2f}")

if __name__ == "__main__":
    # Đường dẫn file đầu vào
    input_file = r"D:\prj\feature\rainfall_30m.tif"
    
    # Đường dẫn file đầu ra
    output_file = r"D:\prj\feature\rainfall_30m.png"
    
    # Chuyển đổi
    tif_to_png(input_file, output_file)
