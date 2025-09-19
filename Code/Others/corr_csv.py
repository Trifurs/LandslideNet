import os
import pandas as pd
import numpy as np

# 定义文件夹路径和结果文件路径
csv_dir = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\correlation"
corr_csv = r"D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\corr1.csv"

# 获取所有 CSV 文件的路径
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

# 初始化一个空的 DataFrame 用于存储最终结果
first_df = pd.read_csv(csv_files[0], index_col=0)
result_df = pd.DataFrame(index=first_df.index, columns=first_df.columns)
result_df.iloc[:, :] = 0  # 将所有值初始化为 0

# 遍历所有 CSV 文件
for csv_file in csv_files:
    df = pd.read_csv(csv_file, index_col=0)
    # 获取数值部分
    values = df.iloc[:, :].values

    # 取绝对值
    abs_values = np.abs(values)

    # 判断是否需要归一化
    if "MutualInfo" in os.path.basename(csv_file):
        min_val = np.min(abs_values)
        max_val = np.max(abs_values)
        if max_val - min_val != 0:
            abs_values = (abs_values - min_val) / (max_val - min_val)

    # 将计算结果累加到结果 DataFrame 中
    result_df.iloc[:, :] += abs_values

# 计算平均值
result_df = result_df / len(csv_files)

# 保存结果到新的 CSV 文件
result_df.to_csv(corr_csv)

print(f"处理完成，结果已保存到: {corr_csv}")