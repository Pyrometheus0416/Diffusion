from pathlib import Path
from typing import TypeAlias

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

Tensor: TypeAlias = torch.Tensor

# TODO: 这个文件专门用于计算 FID 分数，依赖于 InceptionV3 模型提取特征。

# 假设 DEVICE 已定义，如果没有则自动检测
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 定义用于 FID 的专用 Transform
# ==========================================
# 注意：这里不使用 dataset 中的 random crop 或 color jitter
# 必须使用确定性变换，且 Resize 到 299 (InceptionV3 要求)
FID_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)), # 强制 299x299
    transforms.ToTensor(),         # [0, 255] -> [0.0, 1.0] (如果读取的是 uint8) 
    # 如果你的 decode_image 返回的是 [0,1] float，这步可能不需要，但 ToTensor 通常安全
    # InceptionV3 的标准归一化 (ImageNet 统计量)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        # 加载预训练模型
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, 
            aux_logits=False, 
            transform_input=False)
        
        # 移除分类头，只保留 Pool3 输出 (2048维)
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, x: Tensor):
        # x 应该是已经经过 FID_TRANSFORM 处理的 tensor
        # 确保尺寸是 299x299 (虽然 transform 里做了，但双重保险)

        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)

def get_features_from_dataset(dataset_instance, batch_size=32, num_workers=4):
    """
    使用给定的 Dataset 实例提取特征。
    关键：我们需要包装一下原始 dataset，替换掉它的 transform 为 FID_TRANSFORM
    """
    
    # 创建一个包装类，拦截 __getitem__ 并应用正确的 transform
    class FIDWrapper(torch.utils.data.Dataset):
        def __init__(self, original_dataset, transform):
            self.original_dataset = original_dataset
            self.transform = transform
            # 访问原始 dataset 的路径列表 (根据你的类定义)
            # 注意：这里假设 original_dataset.path 是可访问的
            if hasattr(original_dataset, 'path'):
                self.paths = original_dataset.path
            else:
                raise AttributeError("Dataset 没有 'path' 属性，无法直接访问文件列表进行重加载")

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            # 1. 读取原始图像 (复用你原有的 decode_image 逻辑，但不应用原有 transform)
            # 你的原代码: img_t = decode_image(img_path, "RGB").to(device=DEVICE)
            # 为了通用性，我们在这里重新读取，确保拿到 RGB PIL 或 Tensor
            img_path = self.paths[idx]
            
            # 假设 decode_image 是你环境中的一个函数，返回的是 Tensor 或 PIL
            # 如果 decode_image 依赖全局 DEVICE，这里需要注意。
            # 最佳实践是直接读 PIL 然后交给 transform
            from PIL import Image
            img_pil = Image.open(img_path).convert('RGB')
            
            # 2. 应用 FID 专用 Transform (Resize 299 + ImageNet Norm)
            return self.transform(img_pil)

    wrapped_dataset = FIDWrapper(dataset_instance, FID_TRANSFORM)
    loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    extractor = InceptionV3FeatureExtractor()
    features_list = []
    
    print(f"正在提取特征 (共 {len(loader)} 批)...")
    for batch in tqdm(loader):
        batch = batch.to(DEVICE)
        feats = extractor(batch)
        features_list.append(feats.cpu().numpy())
        
    return np.vstack(features_list)

def calculate_fid_score(features_real, features_fake):
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    
    mu_fake = np.mean(features_fake, axis=0)
    sigma_fake = np.cov(features_fake, rowvar=False)
    
    diff = mu_real - mu_fake
    
    # 计算协方差矩阵乘积的平方根
    dot_product = np.dot(sigma_real, sigma_fake)
    sqrt_dot = linalg.sqrtm(dot_product)
    
    if np.iscomplexobj(sqrt_dot):
        sqrt_dot = sqrt_dot.real
        
    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * sqrt_dot)
    return max(0.0, fid)