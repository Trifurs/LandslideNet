import os
import rasterio
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
import logging
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.ops import DeformConv2d  # 正确的导入路径
import shutil
from rasterio.windows import Window


# 加载XML配置文件
def load_config(config_file):
    tree = ET.parse(config_file)
    root = tree.getroot()

    params = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        param_type = param.find('type').text
        if param_type == 'int':
            params[name] = int(value)
        elif param_type == 'float':
            params[name] = float(value)
        elif param_type == 'bool':
            params[name] = True if value.lower() == 'true' else False
        elif param_type == 'list':
            params[name] = str_to_list(value)     
        else:   # folder, file, str
            params[name] = value

    return params

def str_to_list(list_str):
    # 去掉字符串首尾的方括号
    list_str = list_str.strip('[]')
    # 如果字符串为空，直接返回空列表
    if not list_str:
        return []
    # 使用逗号分隔字符串为元素列表
    elements = list_str.split(',')
    # 将每个元素转换为整数
    result = [int(element.strip()) for element in elements]
    return result


class LandslideDataset(Dataset):
    def __init__(self, factors_dir, labels_dir, crop_size=512):
        """
        初始化数据集
        :param factors_dir: 存放影响因子文件的文件夹
        :param labels_dir: 存放标签文件的文件夹
        :param crop_size: 子影像大小（默认为512）
        """
        self.factors_dir = factors_dir
        self.labels_dir = labels_dir
        self.crop_size = crop_size

        # 获取影响因子子文件夹列表（20个影响因子的子文件夹）
        self.factors_subdirs = sorted([d for d in os.listdir(factors_dir) if os.path.isdir(os.path.join(factors_dir, d))])
        
        # 获取标签文件夹（test_cut/labels 下只有一个文件夹）
        self.labels_dir_name = os.listdir(labels_dir)[0]  # 只有一个子文件夹
        self.label_files = sorted([f for f in os.listdir(os.path.join(labels_dir, self.labels_dir_name)) if f.endswith('.tif')])

        # 记录有效样本的索引
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        """
        获取所有包含滑坡点或非滑坡点的影像索引
        """
        valid_indices = []
        for idx, label_file in enumerate(self.label_files):
            label_path = os.path.join(self.labels_dir, self.labels_dir_name, label_file)
            
            # 读取标签影像
            with rasterio.open(label_path) as src:
                label = src.read(1)  # 标签只有一个波段

            # 检查标签中是否有滑坡点和非滑坡点
            if np.any(label == 1) or np.any(label == 2):  # 如果有滑坡点和非滑坡点
                valid_indices.append(idx)
        
        return valid_indices

    def __len__(self):
        # 返回有效样本的数量
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 获取有效样本的真实索引
        real_idx = self.valid_indices[idx]

        # 获取标签文件路径
        label_file = self.label_files[real_idx]
        label_path = os.path.join(self.labels_dir, self.labels_dir_name, label_file)
        
        # 读取标签影像
        with rasterio.open(label_path) as src:
            label = src.read(1)  # 标签只有一个波段

        # 只保留滑坡点和非滑坡点，忽略非样本点和空值
        label = np.where(label == 1, 1, label)  # 滑坡点标记为1
        label = np.where(label == 2, 0, label)  # 非滑坡点标记为0
        label = np.where((label == 0) | (label == 3), -1, label)  # 非样本点和空值标记为-1，代表忽略的区域

        # 读取影响因子的每个影像并堆叠成多波段影像
        factors = []
        for factor_subdir in self.factors_subdirs:
            factor_files = sorted([f for f in os.listdir(os.path.join(self.factors_dir, factor_subdir)) if f.endswith('.tif')])
            factor_file = factor_files[real_idx]  # 获取与标签影像对应的影响因子影像
            
            factor_path = os.path.join(self.factors_dir, factor_subdir, factor_file)
            with rasterio.open(factor_path) as src:
                data = src.read(1)  # 假设每个影像只有一个波段

                # 将数据中不在[0, 1]范围内的值设置为0或1
                data = np.clip(data, 0, 1)  # 将所有小于0的值设为0，大于1的值设为1

                factors.append(data)

        # 将所有影像堆叠成多波段影像
        X = np.stack(factors, axis=0)  # 维度 (num_bands, crop_size, crop_size)

        # 将数据转换为torch tensor
        X = torch.tensor(X, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return X, label

# ##########################################
# # 完整模块
# class LandslideNet(nn.Module):
#     def __init__(self, num_bands, num_classes=2):
#         super(LandslideNet, self).__init__()
        
#         # 编码器部分保持不变
#         self.conv = nn.Conv2d(num_bands, 64, 3, padding=1)
#         self.dcse = DCSELayer(64)
#         self.spb1 = SpatialPerceptionBlock(64, 128)
#         self.spb2 = SpatialPerceptionBlock(128, 256)
#         self.spb3 = SpatialPerceptionBlock(256, 512)
        
#         # 上采样模块
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(512, 256, 3, padding=1)
#         )
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(512, 128, 3, padding=1)
#         )
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(256, 64, 3, padding=1)
#         )
#         self.up4 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(128, num_classes, 3, padding=1)
#         )

#     def forward(self, x):
#         # 编码阶段（记录各阶段输出尺寸）
#         orig_h, orig_w = x.shape[2], x.shape[3]
#         x0 = F.relu(self.dcse(self.conv(x)))       # [B,64,H,W]
#         x0_pool = F.max_pool2d(x0, 2, 2)           # [B,64,H/2,W/2]
        
#         x1 = self.spb1(x0_pool)                    # [B,128,H/4,W/4] 
#         x2 = self.spb2(x1)                         # [B,256,H/8,W/8]
#         x3 = self.spb3(x2)                         # [B,512,H/16,W/16]
        
#         # 步骤1：H/16 → H/8
#         d1 = self.up1(x3)
        
#         # 步骤2：H/8 → H/4（拼接动态调整后的x2）
#         d1 = torch.cat([d1, x2], dim=1)
#         d2 = self.up2(d1)
        
#         # 步骤3：H/4 → H/2（拼接调整后的x1）
#         d2 = torch.cat([d2, x1], dim=1)
#         d3 = self.up3(d2)
        
#         # 步骤4：H/2 → H（拼接调整后的x0_pool）
#         d3 = torch.cat([d3, x0_pool], dim=1)
#         out = self.up4(d3)
        
#         # 最终尺寸校准
#         return F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=True)

# # 保持其他模块不变
# class SpatialPerceptionBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.offset_conv = nn.Conv2d(in_c, 2*3*3, 3, padding=1)
#         self.deform_conv = DeformConv2d(in_c, out_c, 3, padding=1)
#         self.bn = nn.BatchNorm2d(out_c)
#         self.dcse = DCSELayer(out_c)
#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         offsets = self.offset_conv(x)
#         x = F.relu(self.dcse(self.bn(self.deform_conv(x, offsets))))
#         return self.pool(x)

# class DCSELayer(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super().__init__()
#         self.theta = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel//reduction, 1),
#             nn.LayerNorm([channel//reduction, 1, 1]),
#             nn.GELU()
#         )
#         self.phi = nn.Parameter(torch.randn(channel//reduction, channel))
    
#     def forward(self, x):
#         B, C, H, W = x.size()
#         theta = self.theta(x).view(B, -1)
#         phi = F.softmax(self.phi, dim=-1)
#         dynamic_weights = torch.matmul(theta, phi).view(B, C, 1, 1).sigmoid()
#         return x * dynamic_weights.expand_as(x)

###########################################
# SE模块
class LandslideNet(nn.Module):
    def __init__(self, num_bands, num_classes=2):
        super(LandslideNet, self).__init__()
        
        # 加入通道注意力模块处理多源数据
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)
        self.se4 = SELayer(512)

        # 卷积块加入可变形卷积
        self.conv1 = nn.Conv2d(num_bands, 64, 3, padding=1)
        self.conv2 = DeformConvBlock(64, 128)
        self.conv3 = DeformConvBlock(128, 256)
        self.conv4 = DeformConvBlock(256, 512)
        
        # 最后的预测层
        self.final_conv = nn.Conv2d(512, num_classes, 1)
        
    def forward(self, x):
        # 第一层普通卷积
        x = F.relu(self.se1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        
        # 后续层使用可变形卷积
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 最终预测
        x = self.final_conv(x)
        return F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

# 可变形卷积模块（需要单独定义offset生成层）
class DeformConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 生成偏移量的卷积层（3x3卷积核需要18个偏移参数）
        self.offset_conv = nn.Conv2d(in_c, 2*3*3, kernel_size=3, padding=1)  # 输出形状为[b,18,h,w]
        self.deform_conv = DeformConv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 生成偏移量
        offsets = self.offset_conv(x)
        # 执行可变形卷积
        x = F.relu(self.se(self.bn(self.deform_conv(x, offsets))))
        return self.pool(x)

# 通道注意力模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ###########################################
# # CNN模型
# class LandslideNet(nn.Module):
#     def __init__(self, num_bands, num_classes=2):
#         """
#         初始化卷积神经网络模型
#         :param num_bands: 输入影像的波段数
#         :param num_classes: 输出类别数（对于滑坡问题是2，滑坡与非滑坡）
#         """
#         super(LandslideNet, self).__init__()
        
#         # 第1个卷积块
#         self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)  # 输入为num_bands，输出为64通道
#         self.bn1 = nn.BatchNorm2d(64)
        
#         # 第2个卷积块
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出为128通道
#         self.bn2 = nn.BatchNorm2d(128)
        
#         # 第3个卷积块
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 输出为256通道
#         self.bn3 = nn.BatchNorm2d(256)
        
#         # 第4个卷积块
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 输出为512通道
#         self.bn4 = nn.BatchNorm2d(512)

#         # 最后的卷积层输出
#         self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1, padding=0)  # 输出类别数
        
#     def forward(self, x):
#         # 第1个卷积块 + ReLU + 池化
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2, 2)
        
#         # 第2个卷积块 + ReLU + 池化
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2, 2)
        
#         # 第3个卷积块 + ReLU + 池化
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool2d(x, 2, 2)
        
#         # 第4个卷积块 + ReLU + 池化
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.max_pool2d(x, 2, 2)
        
#         # 最后的卷积层，用于输出与输入大小相同的标签图
#         x = self.final_conv(x)
        
#         # 将输出大小恢复到输入大小（通过转置卷积或其他方式）
#         x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)  # 恢复至512x512
        
#         return x

# 数据加载器部分  
def create_dataloaders(factors_dir, labels_dir, batch_size=32, crop_size=512, num_workers=0, seed=20250609):
    # 创建完整数据集
    dataset = LandslideDataset(factors_dir, labels_dir, crop_size)
    total_size = len(dataset)
    
    # 计算各数据集大小（5:2:3）
    train_size = int(0.5 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size  # 自动处理余数问题

    # 设置随机种子保证可重复性
    generator = torch.Generator().manual_seed(seed)
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def create_directories(base_dir, sub_dirs):
    """创建目录结构"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)

def crop_raster(input_raster, output_raster, crop_size=512, overlap=128):
    """裁剪影像并保存"""
    with rasterio.open(input_raster) as src:
        # 获取影像尺寸
        width = src.width
        height = src.height

        # 计算滑动窗口的步长
        step = crop_size - overlap
        
        # 循环进行裁剪
        for i in range(0, height, step):
            for j in range(0, width, step):
                # 设置裁剪窗口
                window = Window(j, i, crop_size, crop_size)
                
                # 关键修正：获取正确的窗口变换矩阵
                transform = src.window_transform(window)
                
                # 读取窗口数据时添加边界处理
                data = src.read(window=window, boundless=True)

                # 为了避免覆盖同名文件，可以在文件名后添加编号或时间戳
                output_file = os.path.join(output_raster, f"{i}_{j}.tif")
                
                # 修正后的写入参数
                with rasterio.open(output_file, 'w', 
                                 driver='GTiff',
                                 height=window.height,
                                 width=window.width,
                                 count=src.count,
                                 dtype=data.dtype,
                                 crs=src.crs,
                                 transform=transform) as dst:
                    dst.write(data)

def crop_all_rasters(input_dir, output_dir, crop_size=512, overlap=128):
    """批量裁剪所有tif影像并保存"""
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    for tif_file in tif_files:
        print(f"Processing {tif_file}...")
        input_raster = os.path.join(input_dir, tif_file)
        output_raster_dir = os.path.join(output_dir, tif_file.replace('.tif', ''))
        create_directories(output_dir, [tif_file.replace('.tif', '')])
        
        crop_raster(input_raster, output_raster_dir, crop_size, overlap)
        print(f"Processing {tif_file} done.")

def create_black_folder(subdir_path):
    """在每个子文件夹下创建black文件夹"""
    black_folder = os.path.join(subdir_path, 'black')
    if not os.path.exists(black_folder):
        os.makedirs(black_folder)
    return black_folder

def collect_black_images(subdir_path):
    """遍历子文件夹中的影像，判断是否为纯黑色影像并收集到列表"""
    black_images = []  # 用于存储所有需要移动的纯黑色影像路径
    
    for file_name in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file_name)
        
        # 只处理tif文件
        if file_name.endswith('.tif'):
            with rasterio.open(file_path) as src:
                # 获取影像数据
                data = src.read(1)  # 假设单波段影像
                
                # 判断影像是否为纯黑色：即影像中是否有绝对值 >= 3 的像素值
                if np.any(np.abs(data) >= 3):  # 如果有任何像素值的绝对值 >= 3
                    black_images.append(file_path)  # 将路径添加到列表
                    # print(f"Identified black image: {file_name} (contains abs value >= 3)")
    
    return black_images

def move_black_images(black_images, black_folder):
    """移动纯黑色影像到black文件夹"""
    for file_path in black_images:
        file_name = os.path.basename(file_path)
        black_file_path = os.path.join(black_folder, file_name)
        shutil.move(file_path, black_file_path)  # 执行移动
        # print(f"Moved black image: {file_name} to {black_folder}")

def move_black_images_in_all_subfolders(output_dir):
    """遍历输出文件夹，检查每个子文件夹中的影像并处理"""
    for subdir_name in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir_name)
        
        # 只处理文件夹
        if os.path.isdir(subdir_path):
            print(f"Processing subfolder: {subdir_name}")
            
            # 创建black文件夹
            black_folder = create_black_folder(subdir_path)
            
            # 收集纯黑色影像的路径
            black_images = collect_black_images(subdir_path)
            
            # 移动纯黑色影像到black文件夹
            if black_images:  # 如果有需要移动的纯黑色影像
                move_black_images(black_images, black_folder)
            print(f"Processing subfolder: {subdir_name} done.")

def collect_image_names_and_paths(subdir_path):
    """遍历子文件夹中的影像，收集影像名称及其路径"""
    image_names = []  # 用于存储影像文件名
    image_paths = []  # 用于存储影像路径
    
    for file_name in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file_name)
        
        # 只处理tif文件
        if file_name.endswith('.tif'):
            image_names.append(file_name)  # 存储文件名
            image_paths.append(file_path)  # 存储文件路径
    
    return image_names, image_paths

def collect_intersection_image_names(output_dir):
    """遍历输出文件夹，收集所有子文件夹中的影像文件名，并求交集"""
    all_image_names = None  # 初始值为None，表示还没有交集
    
    for subdir_name in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir_name)
        
        if os.path.isdir(subdir_path):
            image_names, _ = collect_image_names_and_paths(subdir_path)
            if all_image_names is None:
                all_image_names = set(image_names)  # 第一个子文件夹的文件名作为初始交集
            else:
                all_image_names.intersection_update(image_names)  # 求交集
    
    return all_image_names

def move_to_black_folder(image_path, black_folder):
    """将影像移动到black文件夹"""
    file_name = os.path.basename(image_path)
    black_file_path = os.path.join(black_folder, file_name)
    shutil.move(image_path, black_file_path)  # 执行移动
    print(f"Moved image: {file_name} to {black_folder}")

def move_missing_images_to_black(output_dir):
    """确保所有子文件夹中的影像文件名一致，移动多余的影像到black文件夹"""
    # 收集所有子文件夹中存在的影像文件名的交集
    common_image_names = collect_intersection_image_names(output_dir)
    
    print(f"Common images in all subfolders: {common_image_names}")
    
    for subdir_name in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir_name)
        
        if os.path.isdir(subdir_path):
            print(f"Processing subfolder: {subdir_name}")
            
            # 获取当前子文件夹中的影像名称和路径
            image_names, image_paths = collect_image_names_and_paths(subdir_path)
            
            # 找到当前子文件夹中有，但其他子文件夹没有的影像
            for image_name, image_path in zip(image_names, image_paths):
                if image_name not in common_image_names:  # 如果该影像文件名不在交集中
                    print(f"Image {image_name} is extra, moving to black folder.")
                    # 创建black文件夹
                    black_folder = create_black_folder(subdir_path)
                    # 移动该影像到black文件夹
                    move_to_black_folder(image_path, black_folder)

