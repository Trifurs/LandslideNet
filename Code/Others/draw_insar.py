import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde, pearsonr
import scipy.stats as stats

# 加载数据
csv_path = r'D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\InSAR_POI\R2_slope\POI_slope.csv'
df = pd.read_csv(csv_path)

# 设置图片清晰度和字体大小
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14  # 基础字体大小

# 设置 SCI 论文（英文）风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

# 对数据进行归一化
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

# 选择第三列(index=2)作为 x 变量
x = scaled_df.iloc[:, 2].values.reshape(-1, 1)

# 定义子图尺寸（宽高比一致）
subplot_width = 4.5  # 单个子图宽度（英寸）
subplot_height = 4.0  # 单个子图高度（英寸）
hspace = 0.3  # 子图垂直间距
wspace = 0.25  # 子图水平间距

# 定义第一幅图的列和布局 (2x2)
cols_1 = ['CNN', 'DCSE', 'SPB', 'LandslideNet']
rows_1, cols_1_layout = 2, 2
# 计算第一幅图的总尺寸
fig1_width = cols_1_layout * subplot_width + (cols_1_layout - 1) * wspace
fig1_height = rows_1 * subplot_height + (rows_1 - 1) * hspace

# 绘制第一幅图
fig1, axes1 = plt.subplots(rows_1, cols_1_layout, figsize=(fig1_width, fig1_height))
for i, y_col in enumerate(cols_1):
    row = i // cols_1_layout
    col = i % cols_1_layout

    y = scaled_df[y_col].values

    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # 计算R²
    r2 = r2_score(y, y_pred)
    
    # 计算p值（使用Pearson相关系数）
    corr, p_value = pearsonr(x.flatten(), y)
    
    # 根据p值大小调整显示格式
    if p_value < 1e-10:
        p_text = 'p < 10^{-10}'
    elif p_value < 0.01:
        p_text = f'p = {p_value:.2e}'
    else:
        p_text = f'p = {p_value:.3f}'  # 直接显示三位小数

    # 获取直线方程的系数和截距
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # 组合所有统计信息到一个字符串
    stats_text = f'$\\bf{{y = {slope:.2f}x + {intercept:.2f}}}$\n$\\bf{{R^2 = {r2:.2f}}}$\n$\\bf{{{p_text}}}$'

    # 计算二维核密度估计
    xy = np.vstack([x.flatten(), y])
    z = gaussian_kde(xy)(xy)
    
    # 排序以便绘制平滑的回归线
    sort_idx = np.argsort(x.flatten())
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # 绘制散点图和拟合直线
    scatter = axes1[row, col].scatter(x, y, c=z, s=100, cmap='viridis', alpha=0.7, linewidths=0)
    line = axes1[row, col].plot(x_sorted, y_pred_sorted, color='#FF4B4B', linewidth=5, linestyle='-', alpha=0.8)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes1[row, col], shrink=0.7, aspect=10, pad=0.02)
    cbar.set_label('', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=18, width=1.5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 添加统计信息到右上角
    axes1[row, col].text(0.95, 0.95, stats_text, transform=axes1[row, col].transAxes, 
                         fontsize=20, weight='normal', ha='right', va='top',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.3'),
                         linespacing=1.2)

    # 设置网格线和坐标轴范围
    axes1[row, col].grid(True, linestyle='--', alpha=0.3)
    axes1[row, col].set_xlim([-0.05, 1.05])
    axes1[row, col].set_ylim([-0.05, 1.05])
    
    # 统一设置横纵坐标刻度间隔为0.5
    axes1[row, col].set_xticks(np.arange(0, 1.1, 0.5))
    axes1[row, col].set_yticks(np.arange(0, 1.1, 0.5))
    
    # 坐标轴刻度设置
    axes1[row, col].tick_params(axis='both', which='major', labelsize=18, width=1.5)
    for tick in axes1[row, col].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in axes1[row, col].get_yticklabels():
        tick.set_fontweight('bold')
    
    # 取消显示标签和标题
    axes1[row, col].set_xlabel('')
    axes1[row, col].set_ylabel('')
    axes1[row, col].set_title('')

plt.tight_layout(h_pad=hspace, w_pad=wspace)
plt.savefig(r'D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\InSAR_POI\R2_slope\POI_1.png', dpi=300, bbox_inches='tight')

# 定义第二幅图的列和布局 (2x3)
cols_2 = ['LR', 'CatBoost', 'ET', 'LGBM', 'RF', 'LandslideNet']
rows_2, cols_2_layout = 2, 3
# 计算第二幅图的总尺寸（保持子图尺寸一致）
fig2_width = cols_2_layout * subplot_width + (cols_2_layout - 1) * wspace
fig2_height = rows_2 * subplot_height + (rows_2 - 1) * hspace

# 绘制第二幅图
fig2, axes2 = plt.subplots(rows_2, cols_2_layout, figsize=(fig2_width, fig2_height))
plt.subplots_adjust(hspace=hspace, wspace=wspace)
for i, y_col in enumerate(cols_2):
    row = i // cols_2_layout
    col = i % cols_2_layout

    y = scaled_df[y_col].values

    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # 计算R²
    r2 = r2_score(y, y_pred)
    
    # 计算p值（使用Pearson相关系数）
    corr, p_value = pearsonr(x.flatten(), y)
    
    # 根据p值大小调整显示格式
    if p_value < 1e-10:
        p_text = 'p < 10^{-10}'
    elif p_value < 0.01:
        p_text = f'p = {p_value:.2e}'
    else:
        p_text = f'p = {p_value:.3f}'  # 直接显示三位小数

    # 获取直线方程的系数和截距
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # 组合所有统计信息到一个字符串
    stats_text = f'$\\bf{{y = {slope:.2f}x + {intercept:.2f}}}$\n$\\bf{{R^2 = {r2:.2f}}}$\n$\\bf{{{p_text}}}$'

    # 计算二维核密度估计
    xy = np.vstack([x.flatten(), y])
    z = gaussian_kde(xy)(xy)
    
    # 排序以便绘制平滑的回归线
    sort_idx = np.argsort(x.flatten())
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # 绘制散点图和拟合直线
    scatter = axes2[row, col].scatter(x, y, c=z, s=100, cmap='viridis', alpha=0.7, linewidths=0)
    line = axes2[row, col].plot(x_sorted, y_pred_sorted, color='#FF4B4B', linewidth=5, linestyle='-', alpha=0.8)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes2[row, col], shrink=0.7, aspect=10, pad=0.02)
    cbar.set_label('', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=18, width=1.5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 添加统计信息到右上角
    axes2[row, col].text(0.95, 0.95, stats_text, transform=axes2[row, col].transAxes, 
                         fontsize=20, weight='normal', ha='right', va='top',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.3'),
                         linespacing=1.2)

    # 设置网格线和坐标轴范围
    axes2[row, col].grid(True, linestyle='--', alpha=0.3)
    axes2[row, col].set_xlim([-0.05, 1.05])
    axes2[row, col].set_ylim([-0.05, 1.05])
    
    # 统一设置横纵坐标刻度间隔为0.5
    axes2[row, col].set_xticks(np.arange(0, 1.1, 0.5))
    axes2[row, col].set_yticks(np.arange(0, 1.1, 0.5))
    
    # 坐标轴刻度设置
    axes2[row, col].tick_params(axis='both', which='major', labelsize=18, width=1.5)
    for tick in axes2[row, col].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in axes2[row, col].get_yticklabels():
        tick.set_fontweight('bold')
    
    # 取消显示标签和标题
    axes2[row, col].set_xlabel('')
    axes2[row, col].set_ylabel('')
    axes2[row, col].set_title('')

plt.tight_layout(h_pad=hspace, w_pad=wspace)
plt.savefig(r'D:\lb\myCode\Landslide_susceptibility_mapping\R1_result\InSAR_POI\R2_slope\POI_2.png', dpi=300, bbox_inches='tight')
