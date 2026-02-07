import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import csv
import logging # 引入 logging

# 设置日志系统，用于 LandslideDataset 中的提示
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# 加载XML配置文件 (保持不变)
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
        else:
            params[name] = value
    return params

def str_to_list(list_str):
    list_str = list_str.strip('[]')
    if not list_str:
        return []
    elements = list_str.split(',')
    result = [int(element.strip()) for element in elements]
    return result

# --- 核心修改 1: 针对语义分割任务重写 LandslideDataset ---
class LandslidePatchDataset(Dataset):
    """
    针对 FCN 语义分割模型，加载预处理阶段生成的 .npy 格式 Patch 数据集。
    输入: factors Patch (C, H, W)， 输出: labels Mask (H, W)。
    """
    def __init__(self, factors_base_dir, labels_base_dir, mode='train'):
        """
        初始化数据集
        :param factors_base_dir: output_factors_dir (如: data/processed/factors) 的路径
        :param labels_base_dir: output_labels_dir (如: data/processed/labels) 的路径
        :param mode: 'train', 'val', 或 'test'
        """
        self.factors_dir = os.path.join(factors_base_dir, mode)
        self.labels_dir = os.path.join(labels_base_dir, mode)
        self.samples = []

        # 遍历 '0' (非滑坡) 和 '1' (滑坡) 两个类别文件夹，匹配特征和标签文件
        for class_label in ['0', '1']:
            factors_class_dir = os.path.join(self.factors_dir, class_label)
            labels_class_dir = os.path.join(self.labels_dir, class_label)
            
            if not os.path.exists(factors_class_dir):
                logging.warning(f"Factors directory not found: {factors_class_dir}")
                continue

            # 获取该类别下所有 .npy 文件名 (文件名即为 {r_abs}_{c_abs}.npy)
            # 只需要在 factors 文件夹中查找，假设 labels 文件夹中的文件是匹配的
            file_names = sorted([f for f in os.listdir(factors_class_dir) if f.endswith('.npy')])
            
            for file_name in file_names:
                factor_path = os.path.join(factors_class_dir, file_name)
                label_path = os.path.join(labels_class_dir, file_name)
                
                # 确保标签文件也存在
                if os.path.exists(label_path):
                    self.samples.append((factor_path, label_path))

        if not self.samples:
            raise FileNotFoundError(f"No .npy patches found in {self.factors_dir} and {self.labels_dir}")
        
        logging.info(f"Loaded {len(self.samples)} samples for {mode} set.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        factor_path, label_path = self.samples[idx]

        # 1. 读取影响因子 Patch (C, H, W)
        X = np.load(factor_path) 
        
        # 2. 读取标签 Mask (H, W)
        # label_mask 包含 1 (滑坡点) 和 2 (非滑坡点)，以及其他背景值
        label_mask = np.load(label_path)

        # 3. 标签转换：创建 FCN 训练所需的 Target Mask (H, W)
        # CrossEntropyLoss 期望的类别索引: 类别 1 = 滑坡, 类别 0 = 非滑坡/背景
        # Landslide/Non-Landslide Point TIF 约定: 1=滑坡点, 2=非滑坡点
        # 忽略索引 (ignore_index=-1): 其他值 (如背景, 0, 3) 标记为 -1
        
        # 创建一个初始值为 -1 (忽略) 的 Target Mask
        target = -1 * np.ones_like(label_mask, dtype=np.int64)

        # 将滑坡点 (1) 映射为类别 1
        target[label_mask == 1] = 1 
        
        # 将非滑坡点 (2) 映射为类别 0 
        target[label_mask == 2] = 0
        
        # 转换为 torch tensor
        # X: [C, H, W], dtype=float32
        X = torch.as_tensor(X, dtype=torch.float32)
        # target: [H, W], dtype=torch.long (这是 CrossEntropyLoss 期望的格式)
        target = torch.as_tensor(target, dtype=torch.long) 

        return X, target


##########################################
# --- 辅助模块：1. 优化后的 DCSELayer (保留动态机制, 改进鲁棒性) ---
class DCSELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        # 保留 theta 特征提取路径
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            
            # 优化点 1: 替换 LayerNorm 为 BatchNorm2d
            nn.BatchNorm2d(channel // reduction), 
            
            nn.GELU()
        )
        
        # 保留静态参数 phi
        self.phi = nn.Parameter(torch.randn(channel // reduction, channel))
        
        # 优化点 2: Kaiming 初始化，提高训练鲁棒性
        nn.init.kaiming_uniform_(self.phi, mode='fan_in', nonlinearity='relu') 
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # 1. 动态特征 theta
        theta = self.theta(x).view(B, -1) # [B, C/r]
        
        # 2. 静态权重 phi (使用 softmax 归一化)
        phi = F.softmax(self.phi, dim=-1) # [C/r, C]
        
        # 3. 动态权重计算 (矩阵乘法)
        dynamic_weights = torch.matmul(theta, phi) # [B, C]
        
        # 4. 最终激活和 reshape
        dynamic_weights = dynamic_weights.view(B, C, 1, 1).sigmoid()
        
        # 5. 通道加权
        return x * dynamic_weights.expand_as(x)

# --- 辅助模块：2. 深度可分离卷积 (DSConv) ---
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

# --- 辅助模块：3. SpatialPerceptionBlock (引入残差，移除内部 Pool) ---
class SpatialPerceptionBlock(nn.Module):
    # 保持原 SpatialPerceptionBlock 名称，但功能修改为 ResBlock
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.offset_conv = nn.Conv2d(in_c, 2*3*3, 3, padding=1)
        # 假设 DeformConv2d(in_c, out_c, 3, padding=1) 存在
        self.deform_conv = DeformConv2d(in_c, out_c, 3, padding=1) 
        self.bn = nn.BatchNorm2d(out_c)
        self.dcse = DCSELayer(out_c)
        self.act = nn.ReLU(inplace=True)

        # 残差投影：如果输入输出通道不一致，需要 1x1 卷积调整
        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        
        # --- 主路径 (DeformConv + Attention) ---
        offsets = self.offset_conv(x)
        out = self.deform_conv(x, offsets)
        out = self.bn(out)
        out = self.dcse(out) 
        
        # --- 残差连接 ---
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity # Feature Addition
        out = self.act(out)
        
        return out # 注意：外部 LandslideNet 负责池化

# --- LandslideNet (FCN/U-Net 形式，优化后) ---
class LandslideNet(nn.Module): 
    def __init__(self, num_bands, num_classes=2):
        super(LandslideNet, self).__init__()
        
        # 编码器部分
        self.conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 保持原模块名，但现在功能是 ResBlock
        self.spb1 = SpatialPerceptionBlock(64, 128) 
        self.spb2 = SpatialPerceptionBlock(128, 256) 
        self.spb3 = SpatialPerceptionBlock(256, 512) 
        
        # 下采样池化层 (从 SPB 中移出)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 优化点 3: 引入 Dropout 降低过拟合
        self.dropout = nn.Dropout2d(p=0.2) 
        self.dropout_dec = nn.Dropout2d(p=0.1)
        
        # FPN 风格融合模块 (P5, P4, P3) 
        self.lat_conv3 = nn.Conv2d(512, 256, 1) 
        self.lat_conv2 = nn.Conv2d(256, 256, 1) 
        self.lat_conv1 = nn.Conv2d(128, 256, 1) 
        self.smooth_conv = nn.Conv2d(256, 256, 3, padding=1)
        
        # Decoder: 使用 DSConv 替换原 Conv2d 序列
        # Decoder 1: 256 + 64 (skip) -> 128
        self.dec_conv1 = nn.Sequential(
            DSConv(256 + 64, 128),
            self.dropout_dec, # 轻量级正则化
            DSConv(128, 128)
        )
        
        # Decoder 2: 128 + 64 (skip) -> 64
        self.dec_conv2 = nn.Sequential(
            DSConv(128 + 64, 64),
            self.dropout_dec, # 轻量级正则化
            DSConv(64, 64)
        )

        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        # 1. 初始特征
        x0 = self.conv(x)                   # [B,64,H,W]
        x0_pool = F.max_pool2d(x0, 2, 2)     # [B,64,H/2,W/2]
        
        # 2. 编码阶段 (ResBlock -> Dropout -> Pooling)
        x1_pre = self.spb1(x0_pool)          # [B,128,H/2,W/2]
        x1_pre = self.dropout(x1_pre)
        x1 = self.pool1(x1_pre)              # [B,128,H/4,W/4]
        
        x2_pre = self.spb2(x1)               # [B,256,H/4,W/4]
        x2_pre = self.dropout(x2_pre)
        x2 = self.pool2(x2_pre)              # [B,256,H/8,W/8]
        
        x3_pre = self.spb3(x2)               # [B,512,H/8,W/8]
        x3 = self.pool3(x3_pre)              # [B,512,H/16,W/16] 
        
        # 3. FPN 多尺度融合 (优化: Bilinear)
        c3_lat = self.lat_conv3(x3) 
        c2_lat = self.lat_conv2(x2)
        
        # Bilinear 插值，保证对齐
        c3_up = F.interpolate(c3_lat, size=c2_lat.shape[2:], mode='bilinear', align_corners=True)
        p2 = c3_up + c2_lat                  # [B,256,H/8,W/8]
        
        c1_lat = self.lat_conv1(x1)
        p2_up = F.interpolate(p2, size=c1_lat.shape[2:], mode='bilinear', align_corners=True)
        p1 = p2_up + c1_lat                  # [B,256,H/4,W/4]
        
        # 4. 平滑
        fused_features = self.smooth_conv(p1) # [B,256,H/4,W/4]
        
        # 5. 解码 (优化: Bilinear & DSConv)
        # Stage 1: H/4 -> H/2
        up1 = F.interpolate(fused_features, size=x0_pool.shape[2:], mode='bilinear', align_corners=True)
        concat1 = torch.cat([up1, x0_pool], dim=1)
        dec1 = self.dec_conv1(concat1)
        
        # Stage 2: H/2 -> H
        up2 = F.interpolate(dec1, size=x0.shape[2:], mode='bilinear', align_corners=True)
        concat2 = torch.cat([up2, x0], dim=1)
        dec2 = self.dec_conv2(concat2)
        
        # 6. 输出
        output = self.final_conv(dec2)
        
        return output

# --- 核心修改 2: 适配新的 LandslidePatchDataset 签名 ---
def create_dataloaders(factors_dir, labels_dir, batch_size=32, crop_size=512, num_workers=0, only_test=False):
    """
    创建训练集、验证集和测试集的 DataLoader。
    注意：factors_dir 和 labels_dir 必须分别传入 LandslidePatchDataset。
    
    :param only_test: 如果为 True, 仅返回 test_loader，train/val_loader 返回 None。
    """
    
    # 将 num_workers 设为 4 或 8 (取决于 CPU 核心数) 能够加速数据加载
    num_workers_opt = max(0, int(num_workers)) 
    
    # 始终创建 test_dataset 和 test_loader
    test_dataset = LandslidePatchDataset(factors_dir, labels_dir, mode='test') 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers_opt,
        pin_memory=True
    ) 

    if only_test:
        # 如果只需要测试集，返回 None, None, test_loader
        return None, None, test_loader
    
    # 传入 labels_dir 参数
    train_dataset = LandslidePatchDataset(factors_dir, labels_dir, mode='train')
    val_dataset = LandslidePatchDataset(factors_dir, labels_dir, mode='val')
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers_opt, 
        pin_memory=True, # 优化 CUDA 传输
        drop_last=True
    ) 
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers_opt,
        pin_memory=True
    )
    
    # 正常返回全部三个 loader
    return train_loader, val_loader, test_loader
