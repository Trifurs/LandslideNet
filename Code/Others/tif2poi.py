import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

# 定义输入 tif 文件夹路径和输出 shapefile 文件夹路径
tif_dir = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\LSM_cut\landslide_sensitivity_landslidecnn_mosaic"
shp_dir = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\LSM_cut"

# 确保输出文件夹存在
if not os.path.exists(shp_dir):
    os.makedirs(shp_dir)

# 初始化一个空的 GeoDataFrame 用于存储所有点
points_gdf = gpd.GeoDataFrame(columns=['Num', 'geometry'])

# 获取 tif 文件列表
tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]

# 遍历 tif 文件夹中的所有 tif 文件，并显示进度条
for tif_file in tqdm(tif_files, desc="处理 TIF 文件"):
    # 获取 tif 文件的完整路径
    tif_path = os.path.join(tif_dir, tif_file)
    # 去掉文件扩展名，作为点的 Num 属性值
    num_value = os.path.splitext(tif_file)[0]

    try:
        # 打开 tif 文件
        with rasterio.open(tif_path) as src:
            # 获取栅格的变换信息
            transform = src.transform
            # 获取栅格的宽度和高度
            width = src.width
            height = src.height

            # 计算中间行和中间列
            middle_row = height // 2
            middle_col = width // 2

            # 计算中间像素中心点的空间坐标
            x, y = rasterio.transform.xy(transform, middle_row, middle_col)

            # 创建一个点对象
            point = Point(x, y)

            # 创建一个新的 GeoDataFrame 行
            new_row = gpd.GeoDataFrame({'Num': [num_value], 'geometry': [point]})

            # 将新行添加到总的 GeoDataFrame 中
            points_gdf = gpd.GeoDataFrame(pd.concat([points_gdf, new_row], ignore_index=True))

            # 设置 GeoDataFrame 的坐标系
            points_gdf.crs = src.crs

    except Exception as e:
        print(f"处理文件 {tif_file} 时出错: {e}")

# 保存为 shapefile 文件，并显示进度条
shp_path = os.path.join(shp_dir, 'combined_points.shp')
with tqdm(total=1, desc="保存为 Shapefile") as pbar:
    points_gdf.to_file(shp_path, driver='ESRI Shapefile')
    pbar.update(1)

print(f"处理完成，结果已保存到: {shp_path}")
