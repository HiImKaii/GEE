import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rasterio
import contextily as ctx
import os
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

# ===== ĐỌC FILE TIFF VÀ CHUYỂN ĐỔI HỆ TỌA ĐỘ =====
def read_tiff(file_path):
    with rasterio.open(file_path) as ds:
        data = ds.read(1)  # Đọc band đầu tiên
        bounds = ds.bounds
        src_crs = ds.crs  # Lấy hệ tọa độ gốc
        nodata_value = ds.nodata  # Lấy giá trị NoData từ metadata
        
        print(f"  - NoData value từ file: {nodata_value}")
        print(f"  - Data min: {data.min()}, max: {data.max()}")
        print(f"  - Data dtype: {data.dtype}")
        
        # extent gốc cho vẽ data
        extent_original = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # Chuyển đổi bounds sang WGS84 (EPSG:4326) cho lưới tọa độ
        transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True) if src_crs and src_crs.to_string() != 'EPSG:4326' else None
        if transformer:
            x_min_wgs, y_min_wgs = transformer.transform(bounds.left, bounds.bottom)
            x_max_wgs, y_max_wgs = transformer.transform(bounds.right, bounds.top)
        else:
            x_min_wgs, y_min_wgs, x_max_wgs, y_max_wgs = bounds.left, bounds.bottom, bounds.right, bounds.top
        extent_wgs84 = [x_min_wgs, x_max_wgs, y_min_wgs, y_max_wgs]
    
    return data, extent_original, extent_wgs84, src_crs, nodata_value

# ===== HÀM TẠO BỘ MÀU =====
def get_colormap_for_feature(feature_name):
    """Trả về colormap cho feature"""
    if feature_name == 'esa_worldcover':
        # Màu sắc theo tiêu chuẩn ESA WorldCover
        # Các lớp: 10, 20, 30, 40, 50, 60, 80, 90, 95
        colors = [
            '#006400',  # 10: Tree cover
            '#ffbb22',  # 20: Shrubland
            '#ffff4c',  # 30: Grassland
            '#f096ff',  # 40: Cropland
            '#fa0000',  # 50: Built-up
            '#b4b4b4',  # 60: Bare / sparse vegetation
            '#0064c8',  # 80: Permanent water bodies
            '#00a0a0',  # 90: Herbaceous wetland
            '#00cf75',  # 95: Mangroves
        ]
        labels_dict = {
            10: 'Cây phủ',
            20: 'Cây bụi',
            30: 'Đồng cỏ',
            40: 'Đất trồng trọt',
            50: 'Vùng xây dựng',
            60: 'Đất trống',
            80: 'Nước',
            90: 'Đầm lầy',
            95: 'Rừng ngập mặn'
        }
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, labels_dict, True  # True = dữ liệu phân loại
    elif feature_name == 'lulc':
        # Màu sắc theo tiêu chuẩn LULC với ánh xạ đúng
        colors = [
            '#0064c8',  # 1: Nước (Water bodies) 
            '#006400',  # 2: Cây phủ/Cây bụi (Tree cover/Shrubland)
            '#ffff4c',  # 3: Đồng cỏ (Grassland)
            '#f096ff',  # 4: Đất trồng trọt (Cropland)
            '#fa0000',  # 5: Vùng xây dựng (Built-up)
            '#b4b4b4',  # 6: Đất trống (Bare/sparse vegetation)
        ]
        labels_dict = {
            1: 'Nước',
            2: 'Cây phủ/Cây bụi', 
            3: 'Đồng cỏ',
            4: 'Đất trồng trọt',
            5: 'Vùng xây dựng',
            6: 'Đất trống'
        }
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, labels_dict, True  # True = dữ liệu phân loại
    elif feature_name == 'flow':
        # Màu sắc cho Flow Accumulation - từ thấp đến cao
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', 
                  '#4292c6', '#2171b5', '#08519c', '#08306b']
        label = 'Tích lũy dòng chảy'
        cmap = LinearSegmentedColormap.from_list('flow', colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, label, False  # False = dữ liệu liên tục
    elif feature_name == 'ndvi':
        # Màu sắc cho NDVI - từ không có thực vật đến thực vật dày đặc
        # NDVI: -1 đến 1 (âm = nước, 0 = đất trống, dương = thực vật)
        colors = ['#0000FF', '#8B4513', '#FFFF00', '#90EE90', '#00FF00', '#006400']
        label = 'Chỉ số thực vật NDVI'
        cmap = LinearSegmentedColormap.from_list('ndvi', colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, label, False  # False = dữ liệu liên tục
    elif feature_name == 'twi':
        # Màu sắc cho TWI - màu xanh dương với các bậc màu rõ rệt
        # TWI cao = vùng ẩm ướt, tích nước; TWI thấp = vùng khô, thoát nước tốt
        colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0']
        label = 'Chỉ số độ ẩm địa hình (TWI)'
        cmap = LinearSegmentedColormap.from_list('twi', colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, label, False  # False = dữ liệu liên tục
    elif feature_name == 'curvature':
        # Màu sắc cho Curvature - từ lõm (âm) đến lồi (dương)
        # Âm (lõm/concave) = màu xanh dương, 0 (phẳng) = trắng/vàng nhạt, Dương (lồi/convex) = đỏ/cam
        colors = ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9', '#E1F5FE', '#FFF9C4', '#FFEB3B', '#FFA726', '#FF6F00']
        label = 'Độ cong địa hình (Curvature)'
        cmap = LinearSegmentedColormap.from_list('curvature', colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, label, False  # False = dữ liệu liên tục
    else:
        colors = ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
        label = f'Giá trị {feature_name}'
        cmap = LinearSegmentedColormap.from_list('default', colors)
        cmap.set_bad(color='lightgray', alpha=0.3)
        return cmap, label, False  # False = dữ liệu liên tục

# ===== XỬ LÝ FILE CỤ THỂ =====
input_file = r"D:\prj\feature\gialai_curvature.tif"
output_folder = r"D:\prj\map"

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Đặt tên file output phù hợp với ý nghĩa
output_filename = "ban_do_gialai_curvature.png"
output_file = os.path.join(output_folder, output_filename)

print(f"Đang xử lý file: {input_file}")
print(f"Kết quả sẽ lưu tại: {output_file}\n")

# ===== XỬ LÝ FILE =====
print(f"\n{'='*60}")
print(f"ĐANG XỬ LÝ: Curvature - Độ cong địa hình")
print(f"{'='*60}")  

try:
    print("Đọc file TIFF...")
    data, extent_original, extent_wgs84, src_crs, nodata_value = read_tiff(input_file)
    
    # Curvature là dữ liệu liên tục (continuous), không cần ánh xạ
    print(f"  - Curvature là độ cong địa hình liên tục")
    print(f"    + Giá trị dương (+): Bề mặt lồi (convex), thoát nước tốt")
    print(f"    + Giá trị âm (-): Bề mặt lõm (concave), tích nước")
    print(f"    + Giá trị gần 0: Bề mặt phẳng")
    
    # Mask NoData và NaN
    mask_condition = np.isnan(data)
    if nodata_value is not None:
        mask_condition = mask_condition | (data == nodata_value)
    print(f"  - Masking NaN và NoData values")
    data_masked = np.ma.masked_where(mask_condition, data)

    # Kiểm tra kiểu dữ liệu
    valid_data = data_masked.compressed()  # Lấy dữ liệu không bị mask
    
    if len(valid_data) > 0:
        print(f"Data range (valid): {valid_data.min():.4f} to {valid_data.max():.4f}")
        print(f"Data mean (valid): {valid_data.mean():.4f}")
        print(f"Data std (valid): {valid_data.std():.4f}")
    else:
        print("WARNING: Không có dữ liệu hợp lệ!")
    
    print(f"Data shape: {data.shape}")
    print(f"CRS gốc: {src_crs}")
    print(f"Extent WGS84: {extent_wgs84}")

    # Tạo figure
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('white')

    # Vẽ bản đồ với colormap phù hợp
    print("Vẽ bản đồ...")
    
    # Sử dụng colormap cho Curvature
    cmap_feature, labels_info, is_categorical = get_colormap_for_feature('curvature')

    # Kiểm tra kích thước ảnh để quyết định có thêm basemap không
    total_pixels = data.shape[0] * data.shape[1]
    use_basemap = total_pixels < 100_000_000  # Chỉ dùng basemap nếu < 100M pixels
    
    if use_basemap:
        # Thêm basemap (bản đồ nền) TRƯỚC
        try:
            print("Đang tải bản đồ nền...")
            # Đặt extent cho ax trước
            ax.set_xlim(extent_original[0], extent_original[1])
            ax.set_ylim(extent_original[2], extent_original[3])
            ctx.add_basemap(ax, crs=src_crs.to_string(), 
                            source=ctx.providers.OpenStreetMap.Mapnik, 
                            alpha=0.6, zorder=1, attribution=False)
            print("✓ Đã thêm bản đồ nền")
        except Exception as e:
            print(f"Không thể tải basemap: {e}")
            print("Tiếp tục không có basemap...")
    else:
        print(f"⚠ Bỏ qua basemap do ảnh quá lớn ({data.shape[0]}x{data.shape[1]} = {total_pixels:,} pixels)")

    # Vẽ dữ liệu lên trên
    if is_categorical:
        # Dữ liệu phân loại (ESA WorldCover)
        unique_values = np.unique(valid_data)
        print(f"  - Các lớp có trong dữ liệu: {unique_values}")
        
        # ESA WorldCover có 9 lớp (1-9 sau khi ánh xạ)
        valid_classes = [v for v in unique_values if 1 <= v <= 9]
        print(f"  - Các lớp hiển thị (1-9): {valid_classes}")
        
        # Tính số lượng pixel cho mỗi lớp để thống kê
        print(f"  - Thống kê phân bố:")
        total_valid = len(valid_data)
        
        # Map từ giá trị đã ánh xạ (1-9) về giá trị ESA gốc (10,20,...)
        esa_map = {1:10, 2:20, 3:30, 4:40, 5:50, 6:60, 7:80, 8:90, 9:95}
        
        for val in sorted(valid_classes):
            count = np.sum(valid_data == val)
            percentage = (count / total_valid) * 100
            esa_val = esa_map.get(int(val), int(val))
            label = labels_info.get(esa_val, f'Lớp {esa_val}')
            print(f"    Lớp {esa_val:2d} - {label:20s}: {count:12,} pixels ({percentage:6.2f}%)")
        
        im = ax.imshow(data_masked, extent=extent_original, 
                       cmap=cmap_feature, vmin=0.5, vmax=9.5,
                       aspect='auto', alpha=1.0, zorder=3, interpolation='nearest')
        
        # Colorbar cho dữ liệu phân loại
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7)
        cbar.set_label('Loại sử dụng đất (ESA WorldCover)', fontsize=12, fontweight='bold')
        
        # Đặt ticks và labels cho colorbar - hiển thị các lớp có trong dữ liệu
        cbar.set_ticks(valid_classes)
        cbar.ax.set_yticklabels([labels_info[esa_map[int(i)]] for i in valid_classes], fontsize=9)
    else:
        # Dữ liệu liên tục - Curvature
        if len(valid_data) > 0:
            valid_min = valid_data.min()
            valid_max = valid_data.max()
            
            # Curvature có giá trị âm và dương
            # Sử dụng P5-P95 để giữ dải giá trị rộng (bao gồm cả giá trị 20-30)
            p5 = np.percentile(valid_data, 5)
            p95 = np.percentile(valid_data, 95)
            p25 = np.percentile(valid_data, 25)
            p75 = np.percentile(valid_data, 75)
            median_val = np.median(valid_data)
            
            print(f"  - Dải giá trị thực tế: {valid_min:.2f} to {valid_max:.2f}")
            print(f"  - Median: {median_val:.2f}, Mean: {valid_data.mean():.2f}")
            print(f"  - P5: {p5:.2f}, P25: {p25:.2f}, P75: {p75:.2f}, P95: {p95:.2f}")
            
            # Giữ dải giá trị rộng để bao gồm cả vùng biên (20-30)
            # Làm đối xứng quanh 0
            abs_max = max(abs(p5), abs(p95))
            vmin_plot = -abs_max
            vmax_plot = abs_max
            print(f"  - Sử dụng dải đối xứng quanh 0 (P5-P95): {vmin_plot:.2f} to {vmax_plot:.2f}")
        else:
            vmin_plot = data_masked.min()
            vmax_plot = data_masked.max()
        
        im = ax.imshow(data_masked, extent=extent_original, 
                       cmap=cmap_feature, vmin=vmin_plot, vmax=vmax_plot, 
                       aspect='auto', alpha=1.0, zorder=3, interpolation='bilinear')

        # Colorbar với giá trị min/max và ticks ở khu vực giữa
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7)
        cbar.set_label(labels_info, fontsize=12, fontweight='bold')
        
        # Tạo ticks không đều: dày hơn ở khu vực giữa (-2 đến 2), thưa hơn ở vùng biên
        # Ví dụ: -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30
        tick_values = []
        
        # Vùng âm
        if vmin_plot < -10:
            tick_values.extend([vmin_plot, -20, -10])
        elif vmin_plot < -5:
            tick_values.extend([vmin_plot, -10])
        else:
            tick_values.append(vmin_plot)
        
        # Vùng giữa (dày đặc hơn)
        if vmin_plot < -5:
            tick_values.append(-5)
        if vmin_plot < -2:
            tick_values.append(-2)
        if vmin_plot < -1:
            tick_values.append(-1)
        
        tick_values.append(0)  # Luôn có 0
        
        if vmax_plot > 1:
            tick_values.append(1)
        if vmax_plot > 2:
            tick_values.append(2)
        if vmax_plot > 5:
            tick_values.append(5)
        
        # Vùng dương
        if vmax_plot > 10:
            tick_values.extend([10, 20])
        elif vmax_plot > 5:
            tick_values.append(10)
        
        if vmax_plot > 20:
            tick_values.append(vmax_plot)
        
        # Loại bỏ các giá trị ngoài range
        tick_values = [v for v in tick_values if vmin_plot <= v <= vmax_plot]
        tick_values = sorted(set(tick_values))  # Loại bỏ trùng lặp và sắp xếp
        
        cbar.set_ticks(tick_values)
        cbar.ax.set_yticklabels([f'{val:.1f}' if abs(val) < 10 else f'{val:.0f}' for val in tick_values], fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        
        print(f"  - Colorbar ticks: {tick_values}")

    # Grid tọa độ - Sử dụng WGS84
    lon_range = extent_wgs84[1] - extent_wgs84[0]
    lat_range = extent_wgs84[3] - extent_wgs84[2]
    width_range = extent_original[1] - extent_original[0]
    height_range = extent_original[3] - extent_original[2]

    lon_ticks_wgs = np.linspace(extent_wgs84[0], extent_wgs84[1], 8)
    lat_ticks_wgs = np.linspace(extent_wgs84[2], extent_wgs84[3], 6)
    
    # Chuyển đổi ngược từ WGS84 về hệ tọa độ gốc cho vị trí grid
    transformer_inv = Transformer.from_crs('EPSG:4326', src_crs, always_xy=True) if src_crs and src_crs.to_string() != 'EPSG:4326' else None
    if transformer_inv:
        lon_ticks_orig = [transformer_inv.transform(lon, extent_wgs84[2])[0] for lon in lon_ticks_wgs]
        lat_ticks_orig = [transformer_inv.transform(extent_wgs84[0], lat)[1] for lat in lat_ticks_wgs]
    else:
        lon_ticks_orig = lon_ticks_wgs
        lat_ticks_orig = lat_ticks_wgs

    ax.set_xticks(lon_ticks_orig)
    ax.set_yticks(lat_ticks_orig)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='black', zorder=3)

    # Format nhãn tọa độ - Hiển thị WGS84
    lon_labels = [f'{lon:.3f}°Đ' for lon in lon_ticks_wgs]
    lat_labels = [f'{lat:.3f}°B' for lat in lat_ticks_wgs]
        
    ax.set_xticklabels(lon_labels, fontsize=9)
    ax.set_yticklabels(lat_labels, fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9, top=True, right=True, labeltop=True, labelright=True)

    # North arrow
    x_arrow, y_arrow = 0.95, 0.90
    circle = plt.Circle((x_arrow, y_arrow), 0.025, transform=ax.transAxes, 
                         color='white', ec='black', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    ax.annotate('', xy=(x_arrow, y_arrow+0.02), xytext=(x_arrow, y_arrow-0.01), 
                xycoords='axes fraction', 
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), zorder=11)
    ax.text(x_arrow, y_arrow+0.04, 'N', transform=ax.transAxes, ha='center', va='center', 
            fontsize=14, fontweight='bold', zorder=11)
    ax.text(x_arrow+0.035, y_arrow, 'E', transform=ax.transAxes, ha='center', va='center', 
            fontsize=11, zorder=11)
    ax.text(x_arrow, y_arrow-0.04, 'S', transform=ax.transAxes, ha='center', va='center', 
            fontsize=11, zorder=11)
    ax.text(x_arrow-0.035, y_arrow, 'W', transform=ax.transAxes, ha='center', va='center', 
            fontsize=11, zorder=11)

    # Scale bar - Sử dụng WGS84 cho tính toán km
    km_per_deg = 111.32 * np.cos(np.radians((extent_wgs84[2]+extent_wgs84[3])/2))

    # Vị trí scale bar (dùng extent gốc)
    x_start = extent_original[0] + width_range * 0.05
    y_pos = extent_original[2] + height_range * 0.08

    # Segments và màu - thêm background trắng
    segments = [0, 10, 25, 50]  # km
    bar_colors = ['black', 'white', 'black']
    
    # Tính tỷ lệ chuyển đổi từ km sang đơn vị tọa độ gốc
    scale_factor = width_range / lon_range  # tỷ lệ pixel/degree
    
    # Vẽ nền trắng cho scale bar
    total_width = (segments[-1] / km_per_deg) * scale_factor
    bg_rect = Rectangle((x_start, y_pos - height_range*0.025), 
                        total_width * 1.3, height_range*0.045,
                        facecolor='white', edgecolor='black', linewidth=1.5, zorder=9, alpha=0.9)
    ax.add_patch(bg_rect)

    # Vẽ các đoạn scale bar
    for i in range(len(segments)-1):
        width_km = segments[i+1] - segments[i]
        width = (width_km / km_per_deg) * scale_factor
        rect = Rectangle((x_start + (segments[i]/km_per_deg) * scale_factor, y_pos), 
                         width, height_range*0.01,
                         facecolor=bar_colors[i], edgecolor='black', linewidth=1, zorder=10)
        ax.add_patch(rect)

    # Labels cho scale bar với background trắng
    for km in segments:
        ax.text(x_start + (km/km_per_deg) * scale_factor, 
                y_pos - height_range*0.005, str(km),
                ha='center', va='top', fontsize=10, fontweight='bold', zorder=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    # Label "km"
    ax.text(x_start + (segments[-1]/km_per_deg) * scale_factor * 1.15, 
            y_pos + height_range*0.005, 'km',
            ha='left', va='center', fontsize=11, fontweight='bold', zorder=11)

    # Attribution
    fig.text(0.99, 0.01, '© OpenStreetMap contributors | Dữ liệu: GEE', 
             ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)

    # Labels
    ax.set_xlabel('Kinh độ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vĩ độ', fontsize=12, fontweight='bold')

    # Tiêu đề
    ax.set_title('BẢN ĐỒ ĐỘ CONG ĐỊA HÌNH (CURVATURE)', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    print(f"Lưu file: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Đã tạo bản đồ thành công!")
    
except Exception as e:
    print(f"✗ LỖI khi xử lý: {str(e)}")

print("\n" + "="*60)
print("✓ HOÀN THÀNH!")
print(f"✓ Kết quả lưu tại: {output_file}")
print("="*60)
