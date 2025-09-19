import os
import concurrent.futures
import fiona
import rasterio
from rasterio import features
from rasterio.merge import merge
from rasterio.shutil import copy
from tqdm import tqdm
import numpy as np

# 定义文件夹路径（保持不变）
shp_dir = r"F:\Pakistan\SBAS\Area12\output\cut\graph\S1_Area12_SBAS_SBAS_processing\geocoding\vector"
raster_dir = r"F:\Pakistan\SBAS\Area12\tif"

# 创建存储栅格文件的文件夹
os.makedirs(raster_dir, exist_ok=True)

# 优化1：预定义全局栅格参数
COMMON_RESOLUTION = 0.0001
COMPRESSION = 'DEFLATE'  # 使用压缩减少IO时间
NUM_WORKERS = os.cpu_count() - 2  # 根据CPU核心数设置并行度

def process_single_shp(shp_file):
    """并行处理单个shapefile"""
    try:
        base_name = os.path.splitext(os.path.basename(shp_file))[0]
        output_path = os.path.join(raster_dir, f"{base_name}.tif")
        
        # 优化2：直接使用fiona迭代器避免创建完整GeoDataFrame
        with fiona.open(shp_file) as src:
            # 快速检查属性字段
            if 'velocity' not in src.schema['properties']:
                return None
            
            # 优化3：批量读取几何和属性
            geometries = []
            velocities = []
            for feat in src:
                geometries.append(feat['geometry'])
                velocities.append(feat['properties']['velocity'])
            
            # 获取边界和CRS
            bounds = src.bounds
            crs = src.crs

        # 计算栅格参数
        width = int((bounds[2] - bounds[0]) / COMMON_RESOLUTION)
        height = int((bounds[3] - bounds[1]) / COMMON_RESOLUTION)
        transform = rasterio.transform.from_bounds(*bounds, width, height)
        
        # 优化4：使用更高效的内存布局
        out_arr = features.rasterize(
            zip(geometries, velocities),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.float32
        )

        # 优化5：使用内存文件加速写入
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=transform,
                compress=COMPRESSION  # 添加压缩
            ) as dataset:
                dataset.write(out_arr, 1)
            
            # 将内存文件写入磁盘
            with memfile.open() as src:
                copy(src, output_path, driver='GTiff')
        
        return output_path
    
    except Exception as e:
        print(f"处理文件 {shp_file} 时出错: {str(e)}")
        return None

# 主程序
if __name__ == "__main__":
    # 获取所有shapefile
    shp_files = [os.path.join(shp_dir, f) for f in os.listdir(shp_dir) 
                if f.endswith('.shp')]
    
    # 优化6：使用进程池并行处理
    print("开始并行转换shapefile...")
    raster_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_shp, shp_file): shp_file for shp_file in shp_files}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(shp_files), desc="总进度"):
            result = future.result()
            if result:
                raster_files.append(result)
