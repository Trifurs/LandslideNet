import os
import rasterio
from rasterio.merge import merge
from tqdm import tqdm


def mosaic_rasters(raster_dir, mosaic_file):
    """
    将指定目录下的所有栅格文件镶嵌成一个单独的栅格文件。

    :param raster_dir: 包含待镶嵌栅格文件的目录路径
    :param mosaic_file: 镶嵌后栅格文件的保存路径
    """
    # 获取目录下所有的 tif 文件
    raster_files = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]

    if not raster_files:
        print("指定目录下没有找到 tif 格式的栅格文件，请检查目录路径。")
        return

    # 打开所有的栅格文件
    src_files_to_mosaic = []
    for raster_file in tqdm(raster_files, desc="打开栅格文件"):
        src = rasterio.open(raster_file)
        src_files_to_mosaic.append(src)

    # 执行镶嵌操作
    print("正在进行镶嵌操作...")
    mosaic, out_trans = merge(src_files_to_mosaic)

    # 获取其中一个源文件的元数据，用于输出文件
    out_meta = src.meta.copy()

    # 更新元数据
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # 写入镶嵌后的栅格文件
    print("正在写入镶嵌后的栅格文件...")
    with rasterio.open(mosaic_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # 关闭所有打开的文件
    print("正在关闭所有打开的文件...")
    for src in tqdm(src_files_to_mosaic, desc="关闭栅格文件"):
        src.close()

    print("镶嵌操作完成，结果已保存到：", mosaic_file)


if __name__ == "__main__":
    raster_dir = r"F:\Pakistan\SBAS\Area12\tif"
    mosaic_file = r"F:\Pakistan\SBAS\Area12\mosaic.tif"

    mosaic_rasters(raster_dir, mosaic_file)
