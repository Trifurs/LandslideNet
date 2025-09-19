import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import fiona

# 定义文件路径
shp_file = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\LSM_cut\combined_points.shp"
csv_file = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\corr1.csv"

# 读取 CSV 文件
csv_df = pd.read_csv(csv_file, index_col=0)

# 手动使用 fiona 读取 shapefile
with fiona.open(shp_file) as src:
    shp_gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)

# 获取 CSV 文件首行中第二列到最后一列的列名
new_columns = csv_df.columns

# 为点矢量文件添加新的双精度类型字段
for col in tqdm(new_columns, desc="添加新字段"):
    shp_gdf[col] = None
    shp_gdf[col] = shp_gdf[col].astype('float64')

# 遍历 CSV 文件的每一行，并显示进度条
for index, row in tqdm(csv_df.iterrows(), desc="匹配并赋值", total=len(csv_df)):
    # 去掉行首内容的 .tif 后缀
    num_value = index.replace('.tif', '')

    # 查找点矢量文件中 Num 属性与当前行首内容匹配的点
    matching_points = shp_gdf[shp_gdf['Num'] == num_value]

    # 如果找到匹配的点，则将当前行的值赋给这些点
    if not matching_points.empty:
        for col in new_columns:
            shp_gdf.loc[matching_points.index, col] = row[col]

# 保存更新后的点矢量文件
with tqdm(total=1, desc="保存文件") as pbar:
    shp_gdf.to_file(shp_file)
    pbar.update(1)

print("处理完成，更新后的点矢量文件已保存。")
