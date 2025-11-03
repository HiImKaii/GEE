import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os
from pathlib import Path
import numpy as np

def cut_tiff(input_tiff, shapefile, output_file):
    """
    Cắt một file TIFF theo shapefile
    """
    try:
        # Đọc shapefile
        shapes = gpd.read_file(shapefile)

        # Đọc file TIFF
        with rasterio.open(input_tiff) as src:
            # Lưu lại dtype gốc để giữ nguyên kiểu dữ liệu
            original_dtype = src.dtypes[0]
            
            # Xác định giá trị NoData phù hợp với kiểu dữ liệu
            if src.nodata is not None:
                nodata_value = src.nodata
            else:
                # Chọn giá trị NoData phù hợp với kiểu dữ liệu
                if 'uint8' in original_dtype:
                    nodata_value = 0  # Dùng 0 thay vì 255 cho risk levels
                elif 'uint16' in original_dtype:
                    nodata_value = 0
                elif 'int' in original_dtype:
                    nodata_value = -9999
                else:
                    nodata_value = -9999.0
            
            # Cắt ảnh theo khu vực shapefile với NoData cho vùng bên ngoài
            out_image, out_transform = mask(
                src, 
                shapes.geometry, 
                crop=True,
                nodata=nodata_value,
                filled=True,
                all_touched=False
            )
            out_meta = src.meta.copy()

        # Cập nhật metadata - GIỮ NGUYÊN dtype gốc
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_value,
            "dtype": original_dtype,  # QUAN TRỌNG: Giữ nguyên dtype gốc
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        })

        # Lưu file TIFF đã cắt
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

        # Kiểm tra giá trị unique sau khi cắt
        unique_values = np.unique(out_image[out_image != nodata_value])
        print(f"✓ Cắt thành công: {os.path.basename(input_tiff)} -> {os.path.basename(output_file)}")
        print(f"  Unique values: {unique_values[:20]}")  # In tối đa 20 giá trị để kiểm tra
        return True
    except Exception as e:
        print(f"✗ Lỗi khi cắt {os.path.basename(input_tiff)}: {e}")
        return False

def batch_cut_tiff(input_base_dir, output_base_dir, shapefile):
    """
    Cắt tất cả file TIFF trong thư mục input và lưu vào output với cấu trúc tương tự
    """
    input_path = Path(input_base_dir)
    output_path = Path(output_base_dir)
    
    # Đếm số file đã xử lý
    success_count = 0
    fail_count = 0
    total_count = 0
    
    # Duyệt qua tất cả các file TIFF trong thư mục input
    for tiff_file in input_path.rglob("*.tif"):
        total_count += 1
        
        # Tính toán đường dẫn tương đối từ input_base_dir
        relative_path = tiff_file.relative_to(input_path)
        
        # Tạo đường dẫn output với cấu trúc thư mục tương tự
        output_dir = output_path / relative_path.parent
        
        # Tạo thư mục output nếu chưa tồn tại
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo tên file output với tiền tố "cliped_"
        output_filename = f"cliped_{tiff_file.name}"
        output_file = output_dir / output_filename
        
        print(f"\n[{total_count}] Đang xử lý: {relative_path}")
        
        # Cắt file TIFF
        if cut_tiff(str(tiff_file), shapefile, str(output_file)):
            success_count += 1
        else:
            fail_count += 1
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT:")
    print(f"  Tổng số file: {total_count}")
    print(f"  Thành công: {success_count}")
    print(f"  Thất bại: {fail_count}")
    print("="*60)

if __name__ == "__main__":
    # Đường dẫn thư mục chứa ảnh gốc (unclip)
    input_base_dir = r"D:\prj\results\map\unclip"
    
    # Đường dẫn thư mục lưu ảnh đã cắt (cliped)
    output_base_dir = r"D:\prj\results\map\cliped"

    # Đường dẫn shapefile để cắt
    shapefile = r"C:\Users\Admin\Desktop\GL\gl.shp"
    
    print("BẮT ĐẦU CẮT HÀNG LOẠT ẢNH TIFF")
    print("="*60)
    print(f"Input: {input_base_dir}")
    print(f"Output: {output_base_dir}")
    print(f"Shapefile: {shapefile}")
    print("="*60)
    
    # Kiểm tra xem shapefile có tồn tại không
    if not os.path.exists(shapefile):
        print(f"LỖI: Shapefile không tồn tại: {shapefile}")
    else:
        batch_cut_tiff(input_base_dir, output_base_dir, shapefile)