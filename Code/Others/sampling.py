import os
import numpy as np
import rasterio
from tqdm import tqdm
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
import warnings
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import psutil

warnings.filterwarnings("ignore")

# 配置参数
dtype = np.float32
output_nodata = -9999
base_block_size = 256  # 基础分块大小
n_workers = max(1, cpu_count() - 2)  # 保留2个核心
memmap_threshold = 4  # GB，超过此值启用内存映射

# 配置路径
# 完整数据路径
landslide_path = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_data\Landslide.tif"
factor_folder = r"D:\lb\myCode\Landslide_susceptibility_mapping\Data\origin\factors"
output_folder = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result"

# # 测试数据路径
# landslide_path = r"D:\lb\myCode\Landslide_susceptibility_mapping\Test\Data\origin\label\Landslide_Label.tif"
# factor_folder = r"D:\lb\myCode\Landslide_susceptibility_mapping\Test\Data\origin\factors"
# output_folder = r"D:\lb\myCode\Landslide_susceptibility_mapping\Test\R1_result"

os.makedirs(output_folder, exist_ok=True)

def log_memory_usage(desc):
    """记录内存使用情况"""
    mem = psutil.virtual_memory()
    print(f"{desc} - 内存使用：{mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB")

def auto_block_size(shape):
    """自动计算分块大小"""
    element_size = np.dtype(dtype).itemsize
    available_mem = psutil.virtual_memory().available * 0.5
    max_block = int((available_mem / (shape[2] * element_size)) ** 0.5)
    return min(max_block, base_block_size)

def load_prototype_data():
    """加载滑坡原型数据"""
    with rasterio.open(landslide_path) as src:
        landslide = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
        
        valid_mask = (landslide != nodata) & (~np.isnan(landslide))
        prototype_mask = (landslide == 1) & valid_mask
        
        print(f"有效原型点数量：{np.sum(prototype_mask)}")
        if np.sum(prototype_mask) == 0:
            raise ValueError("未找到有效的滑坡原型点")
            
    return prototype_mask, meta, valid_mask

def create_memmap_array(shape, name):
    """创建内存映射数组"""
    mmap_path = os.path.join(output_folder, f"{name}.dat")
    if os.path.exists(mmap_path):
        os.remove(mmap_path)
    return np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)

def load_environmental_factors():
    """加载并预处理环境因子"""
    factor_files = sorted([f for f in os.listdir(factor_folder) if f.endswith('.tif')])
    
    with rasterio.open(os.path.join(factor_folder, factor_files[0])) as src:
        base = src.read(1)
        rows, cols = base.shape
        transform = src.transform
        crs = src.crs

    # 创建内存映射文件
    factors = create_memmap_array((rows, cols, len(factor_files)), 'factors')
    global_valid = np.ones((rows, cols), dtype=bool)
    
    for idx, f in enumerate(tqdm(factor_files, desc="加载环境因子")):
        with rasterio.open(os.path.join(factor_folder, f)) as src:
            data = src.read(1)
            nodata = src.nodata
            
            # 处理无效值
            valid = (data != nodata) & (~np.isnan(data))
            data[~valid] = np.nan
            
            # 列均值填充
            col_means = np.nanmean(data, axis=0)
            global_col_mean = np.nanmean(col_means)
            col_means = np.where(np.isnan(col_means), global_col_mean, col_means)
            data = np.where(np.isnan(data), col_means[None, :], data)
            
            # 归一化并存储
            factors[:, :, idx] = MinMaxScaler().fit_transform(data.reshape(-1, 1)).reshape(data.shape)
            global_valid &= valid
            
    log_memory_usage("环境因子加载完成")
    return factors, global_valid, transform, crs

def parallel_worker(args):
    """并行计算工作函数"""
    i, j, mmap_path, shape, proto_data, block_size = args
    result = None
    
    try:
        # 加载内存映射数据
        factors = np.memmap(mmap_path, dtype=dtype, mode='r', shape=shape)
        rows, cols, _ = shape
        
        # 计算实际分块范围
        i_end = min(i + block_size, rows)
        j_end = min(j + block_size, cols)
        block = factors[i:i_end, j:j_end, :]
        
        # 处理有效区域
        valid_mask = ~np.isnan(block).any(axis=2)
        if valid_mask.sum() == 0:
            result = np.zeros(block.shape[:2], dtype=dtype)
            return (i, j, i_end, j_end, result)
        
        # 计算KDE
        kde = gaussian_kde(proto_data.T, bw_method='scott')
        flat_data = block[valid_mask].T
        densities = kde.evaluate(flat_data)
        
        # 归一化处理
        max_density = densities.max() if densities.size > 0 else 1.0
        result = np.zeros(block.shape[:2], dtype=dtype)
        result[valid_mask] = densities / (max_density + 1e-8)
        
    except Exception as e:
        print(f"处理块({i},{j})-({i_end},{j_end})失败: {str(e)}")
        result = np.zeros((i_end-i, j_end-j), dtype=dtype)
    finally:
        return (i, j, i_end, j_end, result)

def calculate_similarity(factors, proto_mask):
    """计算相似性分布"""
    # 准备原型数据
    proto_data = factors[proto_mask]
    valid_proto = ~np.isnan(proto_data).any(axis=1)
    proto_data = proto_data[valid_proto].astype(dtype)
    
    if proto_data.size == 0:
        raise ValueError("有效原型数据为空")
    
    print(f"原型数据统计 - 均值: {proto_data.mean(axis=0)}, 标准差: {proto_data.std(axis=0)}")
    
    # 自动调整分块大小
    block_size = auto_block_size(factors.shape)
    print(f"自动分块大小: {block_size}x{block_size}")
    
    # 生成任务队列
    rows, cols, _ = factors.shape
    tasks = [(i, j, factors.filename, factors.shape, proto_data, block_size)
             for i in range(0, rows, block_size)
             for j in range(0, cols, block_size)]
    
    # 初始化结果矩阵
    similarity = np.full((rows, cols), np.nan, dtype=dtype)
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap_unordered(parallel_worker, tasks),
                           total=len(tasks), desc="并行计算相似性"))
    
    # 组装结果
    for i, j, i_end, j_end, block in results:
        similarity[i:i_end, j:j_end] = block
    
    log_memory_usage("相似性计算完成")
    return similarity

def save_raster(data, meta, filename):
    """保存栅格数据"""
    meta.update({
        'dtype': dtype,
        'nodata': output_nodata,
        'count': 1
    })
    
    with rasterio.open(os.path.join(output_folder, filename), 'w', **meta) as dst:
        # 分块写入
        for i in range(0, data.shape[0], base_block_size):
            i_end = min(i + base_block_size, data.shape[0])
            block = data[i:i_end]
            block = np.where(np.isnan(block), output_nodata, block)
            dst.write(block.astype(dtype), 1, window=((i, i_end), (0, data.shape[1])))

def reliability_sampling(similarity, meta):
    """生成可靠性图"""
    landslide_rel = np.clip(similarity, 0, 1)
    non_landslide_rel = 1 - landslide_rel
    
    save_raster(landslide_rel, meta, "landslide_reliability.tif")
    save_raster(non_landslide_rel, meta, "non_landslide_reliability.tif")
    
    return landslide_rel, non_landslide_rel

def apply_global_mask(mask, meta):
    """应用全局掩膜"""
    for fname in os.listdir(output_folder):
        if fname.endswith('.tif'):
            path = os.path.join(output_folder, fname)
            with rasterio.open(path, 'r+') as src:
                data = src.read(1)
                data[~mask] = output_nodata
                src.write(data, 1)

def sample_data(landslide_rel, non_landslide_rel, meta, global_mask, proto_mask, landslide_path):
    """改进后的采样函数，直接使用原始滑坡数据"""
    # 读取原始滑坡数据
    with rasterio.open(landslide_path) as src:
        landslide_data = src.read(1)
        nodata_value = src.nodata
        transform = src.transform
        crs = src.crs

    # 获取正样本坐标（原始滑坡点）
    pos_points = np.argwhere(proto_mask)
    N = len(pos_points)
    print(f"正样本数量: {N}")

    # 获取负样本候选区域（非滑坡且有效区域）
    candidate_mask = (landslide_data != 1) & (landslide_data != nodata_value) & global_mask
    if np.sum(candidate_mask) < N:
        raise ValueError(f"负样本候选点不足，需要{N}个，仅有{np.sum(candidate_mask)}个可用")

    # 随机选择负样本
    np.random.seed(42)
    neg_candidates = np.argwhere(candidate_mask)
    neg_samples = neg_candidates[np.random.choice(len(neg_candidates), N, replace=False)]

    # 创建结果数组
    result = np.full_like(landslide_data, 0, dtype=np.int16)  # 默认填充0
    result[proto_mask] = 1  # 正样本
    result[neg_samples[:, 0], neg_samples[:, 1]] = 2  # 负样本
    
    # 保留原始NoData区域
    result[landslide_data == nodata_value] = nodata_value

    # 生成元数据
    meta.update({
        'dtype': 'int16',
        'nodata': nodata_value,
        'compress': 'lzw'
    })

    # 保存栅格结果
    output_path = os.path.join(output_folder, "final_labels.tif")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(result, 1)
    print(f"标签文件已保存: {output_path}")

    # 生成矢量文件（可选）
    def create_geodataframe(points, label):
        if len(points) == 0:
            return gpd.GeoDataFrame()
        coords = [Point(transform * (j+0.5, i+0.5)) for i, j in points]
        return gpd.GeoDataFrame(geometry=coords, data={'label': [label]*len(points)}, crs=crs)

    # 合并正负样本
    gdf_pos = create_geodataframe(pos_points, 1)
    gdf_neg = create_geodataframe(neg_samples, 2)
    gdf = gpd.GeoDataFrame(pd.concat([gdf_pos, gdf_neg], ignore_index=True))

    # 保存矢量文件
    shp_path = os.path.join(output_folder, "samples.shp")
    if not gdf.empty:
        gdf.to_file(shp_path)
        print(f"矢量样本文件已保存: {shp_path}")

    return result

if __name__ == "__main__":
    try:
        # 1. 加载原型数据
        proto_mask, meta, landslide_valid = load_prototype_data()
        
        # 2. 加载环境因子
        factors, env_valid, transform, crs = load_environmental_factors()
        meta.update({'transform': transform, 'crs': crs})
        
        # 3. 计算全局有效区域
        global_valid = landslide_valid & env_valid
        print(f"全局有效区域比例: {global_valid.mean()*100:.2f}%")
        
        # 4. 计算相似性
        similarity = calculate_similarity(factors, proto_mask)
        
        # 5. 生成可靠性图
        landslide_rel, non_landslide_rel = reliability_sampling(similarity, meta)
        
         # 6. 采样（修改调用参数）
        sample_data(landslide_rel, non_landslide_rel, meta, global_valid, proto_mask, landslide_path)
        
        # 7. 应用掩膜（不再需要，因为采样时已处理NoData）
        apply_global_mask(global_valid, meta)  # 注释掉这行
        
        print("处理完成！结果保存在:", output_folder)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise

