import sys
import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
import torch.nn as nn
from tqdm import tqdm
import glob
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.patheffects import withStroke # 用于文本美化

# 引入您的自定义模块
# 假设 utils.py 包含 LandslideNet 和 create_dataloaders
from utils import LandslideNet, create_dataloaders 

# 过滤警告
warnings.filterwarnings("ignore")

# --- 配置常量 ---
SHAP_NUM_BATCHES = 100
MAX_SUMMARY_POINTS = 200 
FACTOR_VIZ_COLS = 5 
FACTOR_VIZ_ROWS = 4 

# --- 全局 Matplotlib 美化设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['axes.linewidth'] = 1.5 # 增加坐标轴和图例的线条粗细
plt.rcParams['figure.dpi'] = 1000 # 设置默认分辨率

BASE_FONT = 'Times New Roman'

SUMMARY_FONT_CONFIG = {
    'title_size': 25, 'title_weight': 'bold',
    'label_size': 23, 'label_weight': 'bold',
    'tick_size': 21, 'tick_weight': 'bold',
}

GLOBAL_FONT_CONFIG = {
    'title_size': 30, 'title_weight': 'bold',
    'label_size': 28, 'label_weight': 'bold',
    'tick_size': 24, 'tick_weight': 'bold',
    'text_size': 22, 'text_weight': 'bold',
    'legend_title_size': 26, 'legend_title_weight': 'bold',
    'legend_text_size': 24, 'legend_text_weight': 'bold',
}

LOCAL_FONT_CONFIG = {
    'suptitle_size': 32, 'suptitle_weight': 'bold',
    'subplot_title_size': 26, 'subplot_title_weight': 'bold',
    'cbar_label_size': 19, 'cbar_label_weight': 'bold',
}

# --- 字典配置 ---
NAME_DICT = {
    'Slope': 'Slope', 'Aspect': 'Aspect', 'Temperatur': 'Temperature', 'DEM': 'Elevation',
    'Soil': 'Soil', 'Rain': 'Rainfall', 'Earthquake': 'Seismic density', 'Geology': 'Lithology',
    'Dis2Road': 'DTR', 'Dis2River': 'DTV', 'Roughness': 'Roughness', 'Dis2Fault': 'DTF',
    'NDVI': 'NDVI', 'Relief': 'Relief', 'Plan_Curv': 'PLC', 'Profile_Cu': 'PRC',
    'TPI': 'TPI', 'Curv': 'Curvature', 'Population': 'Population', 'TWI': 'TWI'
}

TYPE_DICT = {
    'Slope': 'Topographic', 'Aspect': 'Topographic', 'Temperature': 'Climatic', 'Elevation': 'Topographic',
    'Soil': 'Surficial', 'Rainfall': 'Climatic', 'Seismic density': 'Geological', 'Lithology': 'Geological',
    'DTR': 'Anthropogenic', 'DTV': 'Hydrological', 'Roughness': 'Topographic', 'DTF': 'Geological',
    'NDVI': 'Surficial', 'Relief': 'Topographic', 'PLC': 'Topographic', 'PRC': 'Topographic',
    'TPI': 'Topographic', 'Curvature': 'Topographic', 'Population': 'Anthropogenic', 'TWI': 'Hydrological'
}

# --- 核心模型封装 ---
class LandslideNetWrapper(nn.Module):
    def __init__(self, model, target_class=1):
        super(LandslideNetWrapper, self).__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, x):
        output = self.model(x) 
        target_output = output[:, self.target_class, :, :] 
        # 对输出的空间维度取均值，得到 (Batch, 1) 的张量
        batch_output = torch.mean(target_output, dim=(1, 2)).unsqueeze(1) 
        return batch_output 

# --- 工具函数 ---
def load_factor_names(factors_dir, num_bands):
    file_paths = sorted(glob.glob(os.path.join(factors_dir, '*.tif')))
    if not file_paths:
        file_paths = sorted(glob.glob(os.path.join(factors_dir, '*.tiff')))
    
    if not file_paths:
        return [f"Band_{i+1}" for i in range(num_bands)]

    factor_names = []
    name_lookup = {k.lower(): v for k, v in NAME_DICT.items()}

    for path in file_paths:
        name_with_ext = os.path.basename(path)
        raw_name, _ = os.path.splitext(name_with_ext)
        lookup_key = raw_name.lower()
        
        if lookup_key in name_lookup:
            factor_names.append(name_lookup[lookup_key])
        else:
            clean_key = lookup_key.replace('_', '').replace(' ', '')
            found = False
            for k, v in NAME_DICT.items():
                if k.lower().replace('_', '').replace(' ', '') == clean_key:
                    factor_names.append(v)
                    found = True
                    break
            if not found:
                fallback_name = raw_name
                for k, v in NAME_DICT.items():
                    if k.lower() == lookup_key:
                        fallback_name = v
                        break
                factor_names.append(fallback_name) 
    
    if len(factor_names) > num_bands:
        factor_names = factor_names[:num_bands]
    elif len(factor_names) < num_bands:
        factor_names += [f"Band_{i+1}" for i in range(len(factor_names), num_bands)]
        
    return factor_names

def get_argv(xml_file):
    param_names = [
        'train_output', 'device_ids', 'output_factors_dir', 'output_labels_dir', 
        'num_bands', 'crop_size', 'input_factors_dir' 
    ]
    params = {}
    root = ET.parse(xml_file).getroot()
    for name in param_names:
        for param in root.findall('param'):
            if param.find('name').text == name:
                params[name] = param.find('value').text
                break
        else:
            raise ValueError(f"Parameter {name} not found in XML.")
    return params

def center_crop_tensor(imgs, crop_size):
    if imgs.ndim == 4:
        _, _, h, w = imgs.shape
    elif imgs.ndim == 3:
        _, h, w = imgs.shape
    else:
        raise ValueError(f"Unsupported tensor shape: {imgs.shape}")

    cy, cx = h // 2, w // 2
    sy, sx = crop_size // 2, crop_size // 2
    
    if imgs.ndim == 4:
        return imgs[:, :, cy-sy:cy+sy, cx-sx:cx+sx]
    else: 
        # 修正：当ndim=3时，裁剪应该保留第一维
        return imgs[:, cy-sy:cy+sy, cx-sx:cx+sx] 

def find_slide_sample(test_loader, crop_size, device, max_batches=15):
    full_data_iter = iter(test_loader)
    try:
        background_inputs, _ = next(full_data_iter)
    except StopIteration:
        raise RuntimeError("Test DataLoader is empty.")
        
    background_inputs = center_crop_tensor(background_inputs, crop_size).to(device)
    search_iter = full_data_iter
    
    for _ in tqdm(range(max_batches), desc="Searching Sample", leave=False):
        try:
            current_inputs, current_labels = next(search_iter)
        except StopIteration:
            break
            
        cropped_labels = center_crop_tensor(current_labels, crop_size).cpu().numpy()
        for i in range(len(cropped_labels)):
            if 1 in cropped_labels[i]:
                # 注意：这里需要对 current_inputs 和 current_labels 也进行 center_crop_tensor
                test_inputs = center_crop_tensor(current_inputs, crop_size).to(device)
                test_labels = center_crop_tensor(current_labels, crop_size).to(device)
                return background_inputs, test_inputs, test_labels, i
                
    # 如果找不到滑坡样本，返回第一个批次的第一个样本
    first_batch_inputs, first_batch_labels = next(iter(test_loader))
    test_inputs = center_crop_tensor(first_batch_inputs, crop_size).to(device)
    test_labels = center_crop_tensor(first_batch_labels, crop_size).to(device)
    return background_inputs, test_inputs, test_labels, 0

# --- 绘图函数：所有因子的局部 SHAP 影响图 (已精简刻度并修复裁剪) ---
def plot_local_factor_shap_maps(
    img_tensor, shap_tensor, factor_names, num_bands, output_dir, font_config
):
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    
    factors_to_plot = factor_names[:num_bands]
    if num_bands > FACTOR_VIZ_ROWS * FACTOR_VIZ_COLS:
        factors_to_plot = factors_to_plot[:FACTOR_VIZ_ROWS * FACTOR_VIZ_COLS]
    
    shap_max_abs = np.max(np.abs(shap_tensor)) or 1e-5

    fig, axes = plt.subplots(FACTOR_VIZ_ROWS, FACTOR_VIZ_COLS, 
                             figsize=(FACTOR_VIZ_COLS * 3.5, FACTOR_VIZ_ROWS * 3.8))
    axes = axes.flatten()
    
    print(f"Plotting {len(factors_to_plot)} individual factor SHAP maps...")
    
    # 定义精简的刻度：[-max, 0, max]
    cbar_ticks_raw = np.array([-shap_max_abs, 0.0, shap_max_abs])
    
    for i, name in enumerate(factors_to_plot):
        ax = axes[i]
        
        im = ax.imshow(shap_tensor[i], cmap='seismic', vmin=-shap_max_abs, vmax=shap_max_abs, aspect='auto')
        
        ax.axis('off')
        
        ax.set_title(f"{name}", 
                      fontname=BASE_FONT, 
                      fontweight=font_config['subplot_title_weight'], 
                      fontsize=font_config['subplot_title_size'],
                      pad=10) 
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=1.0)
        
        # 消除颜色条轮廓和空白
        cbar.ax.margins(0) 
        cbar.outline.set_linewidth(0) 
        cbar.ax.tick_params(length=0) # 移除刻度线
        
        # 精简刻度为 [-Max, 0, Max]
        scaled_ticks = cbar_ticks_raw * 1000
        
        cbar.set_ticks(cbar_ticks_raw)
        cbar.set_ticklabels([f'{t:.1f}' for t in scaled_ticks])
        
        # 设置颜色条字体
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(BASE_FONT)
            label.set_fontweight(font_config['cbar_label_weight'])
            label.set_fontsize(font_config['cbar_label_size'])
            
        # 顶部单位标注 (修复裁剪问题，增大x)
        cbar.ax.text(**{'x': 3.5, 'y': 1.05}, 
                      s=r'$\times 10^{-3}$', transform=cbar.ax.transAxes, 
                      fontname=BASE_FONT, 
                      fontweight=font_config['cbar_label_weight'], 
                      fontsize=font_config['cbar_label_size'],
                      ha='center', va='bottom')

    # 隐藏多余的子图
    for j in range(len(factors_to_plot), len(axes)):
        axes[j].axis('off') 
        
    plt.suptitle("Local SHAP Visualization for Individual Factors (Sample Patch)", 
                  fontname=BASE_FONT, 
                  fontweight=font_config['suptitle_weight'], 
                  fontsize=font_config['suptitle_size'],
                  y=0.98) 
    
    # 调整子图间距 (修复裁剪问题，增加右侧边距)
    plt.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.02, wspace=0.3, hspace=0.3)
    
    plt.savefig(os.path.join(output_dir, 'SHAP_Local_Factor_Maps.png'), dpi=300)
    plt.close()


def run_shap_analysis(xml_path):
    print(f"Loading configuration from {xml_path}...")
    params = get_argv(xml_path)
    
    train_output = params['train_output']
    input_factors_dir = params['input_factors_dir'] 
    output_factors_dir = params['output_factors_dir']
    output_labels_dir = params['output_labels_dir']
    num_bands = int(params['num_bands'])
    orig_crop_size = int(params['crop_size']) 
    
    device_ids = list(map(int, params['device_ids'].strip('[]').split(',')))
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    output_dir = os.path.join(os.path.dirname(train_output), 'shap_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    FACTOR_NAMES = load_factor_names(input_factors_dir, num_bands)
    print(f"Factors loaded: {FACTOR_NAMES}")
    
    print("Loading LandslideNet...")
    model = LandslideNet(num_bands=num_bands)
    best_weights = os.path.join(train_output, "best_model_weight.pth")
    if not os.path.exists(best_weights): raise FileNotFoundError(f"Weights not found: {best_weights}")
    model.load_state_dict(torch.load(best_weights, map_location='cpu'))
    model.eval()
    wrapper_model = LandslideNetWrapper(model, target_class=1).to(device)

    BATCH_SIZE = 16 
    _, _, test_loader = create_dataloaders(
        factors_dir=output_factors_dir, labels_dir=output_labels_dir,
        batch_size=BATCH_SIZE, crop_size=orig_crop_size, num_workers=0, only_test=True
    )
    SHAP_CROP = 128 
    
    background_inputs, local_test_inputs, local_test_labels, sample_idx = find_slide_sample(
        test_loader, SHAP_CROP, device, max_batches=15
    )
    
    print("Initializing SHAP for Global Summary...")
    explainer = shap.GradientExplainer(wrapper_model, background_inputs)

    all_shap_values = []
    all_feature_values = []
    multi_batch_loader = iter(test_loader)
    
    processed_batches = 0
    for _ in tqdm(range(SHAP_NUM_BATCHES), desc="Calculating SHAP"):
        try:
            current_inputs, _ = next(multi_batch_loader)
        except StopIteration:
            break
        current_inputs_cropped = center_crop_tensor(current_inputs, SHAP_CROP).to(device)
        shap_values_raw = explainer.shap_values(current_inputs_cropped, ranked_outputs=None)
        
        shap_val = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw
        all_shap_values.append(shap_val)
        all_feature_values.append(current_inputs_cropped.cpu().numpy())
        processed_batches += 1

    if processed_batches == 0: raise RuntimeError("No batches processed.")

    all_shap_values_cat = np.concatenate(all_shap_values, axis=0)
    all_feature_values_cat = np.concatenate(all_feature_values, axis=0)
    B, C, H, W = all_shap_values_cat.shape
    N_total = B * H * W
    shap_values_flat = np.transpose(all_shap_values_cat, (0, 2, 3, 1)).reshape(N_total, C)
    feature_values_flat = np.transpose(all_feature_values_cat, (0, 2, 3, 1)).reshape(N_total, C)

    # --- 绘图 1: SHAP Summary Plot (完全恢复您的精细美化逻辑) ---
    print("\nGenerating SHAP Summary Plot (Violin)...")
    
    if N_total > MAX_SUMMARY_POINTS:
        np.random.seed(42) 
        idx = np.random.choice(N_total, MAX_SUMMARY_POINTS, replace=False)
        shap_sub = shap_values_flat[idx]
        feat_sub = feature_values_flat[idx]
    else:
        shap_sub, feat_sub = shap_values_flat, feature_values_flat
    
    shap_sub_scaled = shap_sub * 10000.0
    
    fig = plt.figure(figsize=(13, 10))
    
    # 使用 violin
    shap.summary_plot(
        shap_sub_scaled, features=feat_sub, feature_names=FACTOR_NAMES[:num_bands],
        max_display=num_bands, show=False, plot_type='violin',
        cmap=plt.cm.RdBu_r
    )
    
    ax = plt.gca()
    
    # ================= 恢复您原本复杂的颜色条控制逻辑 =================
    fig_axes = plt.gcf().axes
    if len(fig_axes) > 1:
        cbar_ax = fig_axes[-1]
        
        pos = cbar_ax.get_position()
        width_multiplier = 3.5 
        new_width = pos.width * width_multiplier
        new_x = pos.x0 - (new_width - pos.width) * 0.5
        
        height_multiplier = 0.85
        new_height = pos.height * height_multiplier
        new_y0 = pos.y0 + (pos.height - new_height) * 0.5
        
        cbar_ax.set_position([new_x, new_y0, new_width, new_height])
        
        main_ax_pos = ax.get_position()
        ax.set_position([main_ax_pos.x0, main_ax_pos.y0, new_x - 0.05, main_ax_pos.height])
        
        texts_to_remove = [text_obj for text_obj in cbar_ax.texts if text_obj.get_text() in ['Low', 'High']]
        for text_obj in texts_to_remove:
            text_obj.remove()

        cbar_ax.set_ylabel('Feature Value', 
                            rotation=90, 
                            labelpad=25, 
                            fontname=BASE_FONT, 
                            fontweight=SUMMARY_FONT_CONFIG['label_weight'],
                            fontsize=SUMMARY_FONT_CONFIG['label_size'])

        cbar_ax.tick_params(labelsize=SUMMARY_FONT_CONFIG['tick_size'])
        for label in cbar_ax.get_yticklabels():
            label.set_fontname(BASE_FONT)
            label.set_fontweight(SUMMARY_FONT_CONFIG['tick_weight'])

    current_ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.1f}' for t in current_ticks], 
                        fontname=BASE_FONT, 
                        fontweight=SUMMARY_FONT_CONFIG['tick_weight'], 
                        fontsize=SUMMARY_FONT_CONFIG['tick_size'])

    plt.yticks(fontname=BASE_FONT, 
               fontweight=SUMMARY_FONT_CONFIG['tick_weight'], 
               fontsize=SUMMARY_FONT_CONFIG['tick_size'])
    
    plt.xlabel('SHAP Value $\\times 10^{-4}$', 
               fontname=BASE_FONT, 
               fontweight=SUMMARY_FONT_CONFIG['label_weight'], 
               fontsize=SUMMARY_FONT_CONFIG['label_size'])
    
    plt.title('SHAP Summary Plot (Violin)', 
              fontname=BASE_FONT, 
              fontweight=SUMMARY_FONT_CONFIG['title_weight'], 
              fontsize=SUMMARY_FONT_CONFIG['title_size'])
    
    plt.savefig(os.path.join(output_dir, 'SHAP_Summary_Plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 绘图 1: SHAP Summary Plot (完全恢复您的精细美化逻辑) ---
    print("\nGenerating SHAP Summary Plot (Violin)...")
    
    if N_total > MAX_SUMMARY_POINTS:
        np.random.seed(42) 
        idx = np.random.choice(N_total, MAX_SUMMARY_POINTS, replace=False)
        shap_sub = shap_values_flat[idx]
        feat_sub = feature_values_flat[idx]
    else:
        shap_sub, feat_sub = shap_values_flat, feature_values_flat
    
    shap_sub_scaled = shap_sub * 10000.0
    
    # 修复点：定义 SCI 经典的蓝紫红配色
    # 如果 shap.colors 报错，通常可以使用内置的字符串 'coolwarm' 或手动获取 SHAP 颜色
    import matplotlib.colors as mcolors
    # 这是 SHAP 经典的蓝到红的颜色定义
    shap_red_blue = mcolors.LinearSegmentedColormap.from_list("shap_red_blue", ["#1E88E5", "#ff0052"])

    fig = plt.figure(figsize=(13, 10))
    
    # 使用 violin
    shap.summary_plot(
        shap_sub_scaled, features=feat_sub, feature_names=FACTOR_NAMES[:num_bands],
        max_display=num_bands, show=False, plot_type='violin',
        cmap=shap_red_blue # 使用修复后的 cmap
    )
    
    ax = plt.gca()
    
    # ================= 恢复您原本复杂的颜色条控制逻辑 =================
    fig_axes = plt.gcf().axes
    if len(fig_axes) > 1:
        cbar_ax = fig_axes[-1]
        
        pos = cbar_ax.get_position()
        width_multiplier = 3.5 
        new_width = pos.width * width_multiplier
        new_x = pos.x0 - (new_width - pos.width) * 0.5
        
        height_multiplier = 0.85
        new_height = pos.height * height_multiplier
        new_y0 = pos.y0 + (pos.height - new_height) * 0.5
        
        cbar_ax.set_position([new_x, new_y0, new_width, new_height])
        
        main_ax_pos = ax.get_position()
        ax.set_position([main_ax_pos.x0, main_ax_pos.y0, new_x - 0.05, main_ax_pos.height])
        
        texts_to_remove = [text_obj for text_obj in cbar_ax.texts if text_obj.get_text() in ['Low', 'High']]
        for text_obj in texts_to_remove:
            text_obj.remove()

        cbar_ax.set_ylabel('Feature Value', 
                            rotation=90, 
                            labelpad=25, 
                            fontname=BASE_FONT, 
                            fontweight=SUMMARY_FONT_CONFIG['label_weight'],
                            fontsize=SUMMARY_FONT_CONFIG['label_size'])

        cbar_ax.tick_params(labelsize=SUMMARY_FONT_CONFIG['tick_size'])
        for label in cbar_ax.get_yticklabels():
            label.set_fontname(BASE_FONT)
            label.set_fontweight(SUMMARY_FONT_CONFIG['tick_weight'])

    current_ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.1f}' for t in current_ticks], 
                        fontname=BASE_FONT, 
                        fontweight=SUMMARY_FONT_CONFIG['tick_weight'], 
                        fontsize=SUMMARY_FONT_CONFIG['tick_size'])

    plt.yticks(fontname=BASE_FONT, 
               fontweight=SUMMARY_FONT_CONFIG['tick_weight'], 
               fontsize=SUMMARY_FONT_CONFIG['tick_size'])
    
    plt.xlabel('SHAP Value $\\times 10^{-4}$', 
               fontname=BASE_FONT, 
               fontweight=SUMMARY_FONT_CONFIG['label_weight'], 
               fontsize=SUMMARY_FONT_CONFIG['label_size'])
    
    plt.title('SHAP Summary Plot (Violin)', 
              fontname=BASE_FONT, 
              fontweight=SUMMARY_FONT_CONFIG['title_weight'], 
              fontsize=SUMMARY_FONT_CONFIG['title_size'])
    
    plt.savefig(os.path.join(output_dir, 'SHAP_Summary_Plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 新增: SHAP Summary Plot (Beeswarm) ---
    print("Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(13, 10))
    
    shap.summary_plot(
        shap_sub_scaled, features=feat_sub, feature_names=FACTOR_NAMES[:num_bands], 
        plot_type='dot', show=False, cmap=shap_red_blue # 保持配色一致
    )
    
    ax = plt.gca()

    # 1. 调整散点大小
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.collections.PathCollection):
            child.set_sizes([25]) 

    # 2. 统一字体和格式
    current_ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.1f}' for t in current_ticks], 
                        fontname=BASE_FONT, 
                        fontweight=SUMMARY_FONT_CONFIG['tick_weight'], 
                        fontsize=SUMMARY_FONT_CONFIG['tick_size'])

    plt.yticks(fontname=BASE_FONT, 
               fontweight=SUMMARY_FONT_CONFIG['tick_weight'], 
               fontsize=SUMMARY_FONT_CONFIG['tick_size'])
    
    plt.xlabel('SHAP Value $\\times 10^{-4}$', 
               fontname=BASE_FONT, 
               fontweight=SUMMARY_FONT_CONFIG['label_weight'], 
               fontsize=SUMMARY_FONT_CONFIG['label_size'])
    
    plt.title('SHAP Summary Plot (Beeswarm)', 
              fontname=BASE_FONT, 
              fontweight=SUMMARY_FONT_CONFIG['title_weight'], 
              fontsize=SUMMARY_FONT_CONFIG['title_size'])

    # 3. 颜色条美化
    fig_axes = plt.gcf().axes
    if len(fig_axes) > 1:
        cbar_ax = fig_axes[-1]
        for t in cbar_ax.texts:
            if t.get_text() in ['Low', 'High']:
                t.set_visible(False)
        
        cbar_ax.set_ylabel('Feature Value', 
                           fontname=BASE_FONT,
                           fontweight=SUMMARY_FONT_CONFIG['label_weight'],
                           fontsize=SUMMARY_FONT_CONFIG['label_size'],
                           labelpad=15)
        
        cbar_ax.tick_params(labelsize=SUMMARY_FONT_CONFIG['tick_size'])
        for label in cbar_ax.get_yticklabels():
            label.set_fontname(BASE_FONT)
            label.set_fontweight(SUMMARY_FONT_CONFIG['tick_weight'])

    plt.savefig(os.path.join(output_dir, 'SHAP_Summary_Beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 绘图 2: Global Importance with Categories (完全恢复您的代码) ---
    print("Generating Classified Global Importance Plot...")
    
    global_imp = np.mean(np.abs(shap_values_flat), axis=0)
    global_imp_norm = global_imp / np.sum(global_imp)
    
    indices = np.argsort(global_imp_norm)[::-1]
    sorted_names = [FACTOR_NAMES[i] for i in indices]
    sorted_values = global_imp_norm[indices]
    
    unique_types = sorted(list(set(TYPE_DICT.values())))
    base_colors = plt.cm.get_cmap('tab10', len(unique_types))
    type_color_map = {t: base_colors(i) for i, t in enumerate(unique_types)}
    type_color_map['Unknown'] = (0.5, 0.5, 0.5, 1.0)
    
    bar_colors = []
    for name in sorted_names:
        category = TYPE_DICT.get(name, 'Unknown')
        bar_colors.append(type_color_map[category])

    plt.figure(figsize=(14, 10))
    ax_bar = plt.gca()
    
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    
    plt.barh(range(len(sorted_values)), sorted_values, color=bar_colors, align='center', zorder=3)
    
    plt.yticks(range(len(sorted_values)), sorted_names, 
               fontname=BASE_FONT, 
               fontweight=GLOBAL_FONT_CONFIG['tick_weight'], 
               fontsize=GLOBAL_FONT_CONFIG['tick_size'])
    
    # === 修复点：设置横坐标刻度值的字体 ===
    plt.xticks(fontname=BASE_FONT, 
               fontweight=GLOBAL_FONT_CONFIG['tick_weight'], 
               fontsize=GLOBAL_FONT_CONFIG['tick_size'])
    
    plt.xlabel('Mean Absolute SHAP Value (Normalized)', 
               fontname=BASE_FONT, 
               fontweight=GLOBAL_FONT_CONFIG['label_weight'], 
               fontsize=GLOBAL_FONT_CONFIG['label_size'])
    
    plt.title('Global Feature Importance by Category', 
              fontname=BASE_FONT, 
              fontweight=GLOBAL_FONT_CONFIG['title_weight'], 
              fontsize=GLOBAL_FONT_CONFIG['title_size'])
              
    ax_bar.invert_yaxis()
    
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)
    
    for i, v in enumerate(sorted_values):
        plt.text(v + 0.002, i, f'{v*100:.1f}%', va='center', 
                 fontname=BASE_FONT, 
                 fontsize=GLOBAL_FONT_CONFIG['text_size'], 
                 fontweight=GLOBAL_FONT_CONFIG['text_weight'],
                 path_effects=[withStroke(linewidth=1, foreground='white')])

    legend_patches = [mpatches.Patch(color=type_color_map[t], label=t) for t in unique_types]
    if 'Unknown' in [TYPE_DICT.get(n, 'Unknown') for n in sorted_names]:
        legend_patches.append(mpatches.Patch(color=type_color_map['Unknown'], label='Unknown'))
        
    plt.legend(handles=legend_patches, title='Factor Category', 
               frameon=True, 
               facecolor='white', 
               edgecolor='gray', 
               fancybox=True, 
               prop={'family': BASE_FONT, 
                     'size': GLOBAL_FONT_CONFIG['legend_text_size'], 
                     'weight': GLOBAL_FONT_CONFIG['legend_text_weight']},
               title_fontproperties={'family': BASE_FONT, 
                                     'weight': GLOBAL_FONT_CONFIG['legend_title_weight'], 
                                     'size': GLOBAL_FONT_CONFIG['legend_title_size']})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SHAP_Global_Importance.png'), dpi=300)
    plt.close()
    
    with open(os.path.join(output_dir, 'SHAP_Global_Importance.csv'), 'w') as f:
        f.write("Rank,Factor,Category,Importance\n")
        for i, (name, val) in enumerate(zip(sorted_names, sorted_values)):
            cat = TYPE_DICT.get(name, 'Unknown')
            f.write(f"{i+1},{name},{cat},{val:.6f}\n")

    # --- 新增: Dependence Plot & Interaction Effect ---
    print("Generating Dependence Plots...")
    for i in range(min(2, len(indices))):
        idx_f = indices[i]
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(idx_f, shap_values_flat, feature_values_flat, feature_names=FACTOR_NAMES, show=False)
        plt.savefig(os.path.join(output_dir, f'SHAP_Dependence_{FACTOR_NAMES[idx_f]}.png'), dpi=300)
        plt.close()

    # --- 新增: Waterfall/Force/Decision (解决 expected_value 报错) ---
    print("Generating Local Sample Explanations...")
    # 获取基准值 (适配不同版本的属性名)
    if hasattr(explainer, 'expected_value'):
        base_val = explainer.expected_value
    else:
        # 备选方案：通过模型在空输入上的预测估算（GradientExplainer通常有此属性）
        base_val = 0.0 
    
    if isinstance(base_val, (list, np.ndarray, torch.Tensor)):
        base_val = base_val[0]

    # 提取滑坡样本中心像素点
    ch, cw = SHAP_CROP // 2, SHAP_CROP // 2
    px_shap = all_shap_values_cat[sample_idx, :, ch, cw]
    px_feat = all_feature_values_cat[sample_idx, :, ch, cw]

    # Waterfall
    plt.figure(figsize=(12, 8))
    shap.plots._waterfall.waterfall_legacy(base_val, px_shap, feature_names=FACTOR_NAMES, show=False)
    plt.savefig(os.path.join(output_dir, 'SHAP_Waterfall.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Force Plot
    plt.figure(figsize=(20, 3))
    shap.force_plot(base_val, px_shap, np.around(px_feat, 2), feature_names=FACTOR_NAMES, matplotlib=True, show=False)
    plt.savefig(os.path.join(output_dir, 'SHAP_Force_Plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Decision Plot
    plt.figure(figsize=(10, 8))
    shap.decision_plot(base_val, px_shap, feature_names=FACTOR_NAMES, show=False)
    plt.savefig(os.path.join(output_dir, 'SHAP_Decision_Plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 绘图 3: 所有因子的局部可视化 (恢复您的代码) ---
    print("Generating Local Factor SHAP Maps...")
    
    local_shap_raw = explainer.shap_values(local_test_inputs, ranked_outputs=None)
    shap_val_local = local_shap_raw[0] if isinstance(local_shap_raw, list) else local_shap_raw
    
    img_tensor = local_test_inputs[sample_idx].cpu().numpy() 
    shap_tensor = shap_val_local[sample_idx] 
    
    plot_local_factor_shap_maps(
        img_tensor, shap_tensor, FACTOR_NAMES, num_bands, output_dir, LOCAL_FONT_CONFIG
    )
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2: raise RuntimeError("Missing XML config file parameter")
        run_shap_analysis(sys.argv[1])
        print('<shap_status>0</shap_status>')
        print('<shap_log>success</shap_log>')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('<shap_status>1</shap_status>')
        print(f'<shap_log>{str(e)}</shap_log>')
