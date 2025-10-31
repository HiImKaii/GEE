import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os

def cut_tiff(input_tiff, shapefile, output_dir):
    try:
        # Đọc shapefile
        shapes = gpd.read_file(shapefile)

        # Đọc file TIFF
        with rasterio.open(input_tiff) as src:
            # Xác định giá trị NoData phù hợp với kiểu dữ liệu
            if src.nodata is not None:
                nodata_value = src.nodata
            else:
                # Chọn giá trị NoData phù hợp với kiểu dữ liệu
                dtype = src.dtypes[0]
                if 'uint8' in dtype:
                    nodata_value = 255
                elif 'uint16' in dtype:
                    nodata_value = 65535
                elif 'int' in dtype:
                    nodata_value = -9999
                else:
                    nodata_value = -9999.0
            
            # Cắt ảnh theo khu vực shapefile với NoData cho vùng bên ngoài
            out_image, out_transform = mask(
                src, 
                shapes.geometry, 
                crop=True,
                nodata=nodata_value,
                filled=False
            )
            out_meta = src.meta

        # Cập nhật metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_value
        })

        # Tạo đường dẫn file đầu ra
        output_file = os.path.join(output_dir, os.path.basename(input_tiff))

        # Lưu file TIFF đã cắt
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Cắt ảnh thành công! File đầu ra: {output_file}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    input_tiff = r"C:\Users\Admin\Downloads\ESA_WorldCover_30m_2021.tif"
    shapefile = r"C:\Users\Admin\Desktop\GL\gl.shp"
    output_dir = r"D:"

    cut_tiff(input_tiff, shapefile, output_dir)