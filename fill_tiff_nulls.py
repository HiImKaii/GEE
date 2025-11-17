"""
Fill null values in TIFF files using neighboring pixel values.
Gradually fills from the boundary inward, constrained by a shapefile boundary.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import geometry_mask
from scipy import ndimage
import geopandas as gpd
import os


def fill_null_values(input_path, output_path, shapefile_path=None, nodata_value=None, max_iterations=1000):
    """
    Fill null/nodata values in a raster using neighboring valid pixels.
    Fills gradually from the edge inward, constrained by shapefile boundary.
    
    Parameters:
    -----------
    input_path : str
        Path to input TIFF file
    output_path : str
        Path to output TIFF file
    shapefile_path : str, optional
        Path to shapefile defining the valid boundary area. 
        Only pixels within this boundary will be filled.
    nodata_value : float, optional
        Value representing nodata. If None, will use the nodata value from the file metadata.
    max_iterations : int
        Maximum number of iterations for filling
    """
    
    # Read the input raster
    with rasterio.open(input_path) as src:
        data = src.read(1)  # Read first band
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        
        # Get nodata value from file if not specified
        if nodata_value is None:
            nodata_value = src.nodata
            if nodata_value is None:
                nodata_value = np.nan
    
    print(f"Input shape: {data.shape}")
    print(f"Nodata value: {nodata_value}")
    
    # Create boundary mask from shapefile if provided
    boundary_mask = None
    if shapefile_path:
        print(f"Loading shapefile boundary: {shapefile_path}")
        try:
            # Read shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Reproject to match raster CRS if needed
            if gdf.crs != crs:
                print(f"Reprojecting shapefile from {gdf.crs} to {crs}")
                gdf = gdf.to_crs(crs)
            
            # Create mask from geometries (True = inside boundary, False = outside)
            boundary_mask = ~geometry_mask(
                gdf.geometry,
                out_shape=data.shape,
                transform=transform,
                invert=False
            )
            
            pixels_in_boundary = np.sum(boundary_mask)
            print(f"Pixels within shapefile boundary: {pixels_in_boundary}")
            
        except Exception as e:
            print(f"Warning: Could not load shapefile: {str(e)}")
            print("Continuing without boundary constraint...")
            boundary_mask = None
    
    # Create mask for pixels that have data (not nodata AND not NaN)
    # Handle both nodata value and NaN
    if np.isnan(nodata_value):
        has_data_mask = ~np.isnan(data)
    else:
        # Check for both nodata value AND NaN
        has_data_mask = (data != nodata_value) & (~np.isnan(data))
    
    print(f"Pixels with data: {np.sum(has_data_mask)}")
    print(f"NaN pixels: {np.sum(np.isnan(data))}")
    if not np.isnan(nodata_value):
        print(f"NoData ({nodata_value}) pixels: {np.sum(data == nodata_value)}")
    
    # Determine which pixels need to be filled based on shapefile
    if boundary_mask is not None:
        # Pixels that SHOULD have data = inside shapefile boundary
        should_have_data = boundary_mask
        
        # Pixels that are NULL = inside boundary BUT don't have data
        null_mask = should_have_data & (~has_data_mask)
        
        # Valid mask = pixels that currently have valid data
        mask = has_data_mask.copy()
        
        print(f"Pixels inside shapefile boundary: {np.sum(should_have_data)}")
        print(f"Pixels with data inside boundary: {np.sum(should_have_data & has_data_mask)}")
        print(f"NULL pixels to fill (in boundary but no data): {np.sum(null_mask)}")
        
        fillable_area = should_have_data
        null_count = np.sum(null_mask)
    else:
        # No boundary constraint - use original logic
        mask = has_data_mask
        fillable_area = np.ones_like(mask, dtype=bool)
        null_mask = ~mask
        null_count = np.sum(null_mask)
        print(f"Null pixels to fill: {null_count}")
    
    if null_count == 0:
        print("No null values to fill!")
        return
    
    # Create a copy of the data to modify
    filled_data = data.copy()
    
    # Define 8-neighbor kernel for checking adjacent pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Iteratively fill null values
    iteration = 0
    filled_count = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Find null pixels that have at least one valid neighbor
        # Dilate the valid mask to find border pixels
        dilated_mask = ndimage.binary_dilation(mask, structure=kernel)
        
        # Border pixels are those that are null but adjacent to valid pixels
        # AND within the fillable area
        border_pixels = dilated_mask & (~mask) & fillable_area
        
        if not np.any(border_pixels):
            print(f"No more border pixels to fill after {iteration} iterations")
            break
        
        # Get indices of border pixels
        border_indices = np.where(border_pixels)
        
        # Fill each border pixel with the mean of its valid neighbors
        for i, j in zip(border_indices[0], border_indices[1]):
            # Get neighborhood
            i_min = max(0, i - 1)
            i_max = min(data.shape[0], i + 2)
            j_min = max(0, j - 1)
            j_max = min(data.shape[1], j + 2)
            
            neighborhood = filled_data[i_min:i_max, j_min:j_max]
            neighborhood_mask = mask[i_min:i_max, j_min:j_max]
            
            # Calculate mean of valid neighbors
            if np.any(neighborhood_mask):
                valid_values = neighborhood[neighborhood_mask]
                filled_data[i, j] = np.mean(valid_values)
                mask[i, j] = True
                filled_count += 1
        
        if iteration % 10 == 0:
            remaining = null_count - filled_count
            print(f"Iteration {iteration}: Filled {filled_count}/{null_count} pixels, {remaining} remaining")
    
    print(f"Filling complete! Total pixels filled: {filled_count}")
    
    # Update profile for output
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=nodata_value if not np.isnan(nodata_value) else None
    )
    
    # Write output
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filled_data.astype(rasterio.float32), 1)
    
    print(f"Output saved to: {output_path}")


def process_directory(input_dir, output_dir, shapefile_path=None, nodata_value=None, max_iterations=1000, recursive=False):
    """
    Process all TIFF files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input TIFF files
    output_dir : str
        Directory for output TIFF files
    shapefile_path : str, optional
        Path to shapefile defining the valid boundary area
    nodata_value : float, optional
        Value representing nodata
    max_iterations : int
        Maximum number of iterations for filling
    recursive : bool
        If True, search for TIFF files in subdirectories as well
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIFF files
    tiff_extensions = ['.tif', '.tiff']
    tiff_files = []
    
    if recursive:
        # Search recursively in subdirectories
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f.lower())[1] in tiff_extensions:
                    rel_path = os.path.relpath(root, input_dir)
                    tiff_files.append((os.path.join(root, f), rel_path, f))
    else:
        # Only search in the specified directory
        for f in os.listdir(input_dir):
            full_path = os.path.join(input_dir, f)
            if os.path.isfile(full_path) and os.path.splitext(f.lower())[1] in tiff_extensions:
                tiff_files.append((full_path, '', f))
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    
    for file_info in tiff_files:
        input_path, rel_path, filename = file_info
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.join(rel_path, filename) if rel_path else filename}")
        print(f"{'='*60}")
        
        # Create subdirectory structure in output directory if needed
        if rel_path:
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
        else:
            output_subdir = output_dir
        
        output_filename = os.path.splitext(filename)[0] + '_filled' + os.path.splitext(filename)[1]
        output_path = os.path.join(output_subdir, output_filename)
        
        try:
            fill_null_values(input_path, output_path, shapefile_path, nodata_value, max_iterations)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Shapefile path for boundary constraint
    shapefile_boundary = r"C:\Users\Admin\Desktop\GL\gl.shp"
    
    # Example usage - process single file
    # input_file = r"D:\prj\results\map\cliped\rf\cliped_flood_probability_pso_RF.tif"
    # output_file = r"D:\prj\results\map\cliped\filled\cliped_flood_probability_pso_RF_filled.tif"
    # fill_null_values(input_file, output_file, shapefile_boundary)
    
    # Process all files in directory and subdirectories
    input_directory = r"D:\prj\results\map\cliped"
    output_directory = r"D:\prj\results\map\cliped\filled"
    
    # You can specify nodata value if needed, or set to None to auto-detect
    # Common nodata values: -9999, -3.4028235e+38, np.nan
    process_directory(
        input_directory, 
        output_directory,
        shapefile_path=shapefile_boundary,  # Use shapefile to constrain fill area
        nodata_value=None,  # Auto-detect from file
        max_iterations=1000,  # Increase if needed for large gaps
        recursive=True  # Search in subdirectories (rf, svr, xgb)
    )
