"""
第一阶段 Transformer + CNN 推理脚本
输出: CSV 格式，配合复验工具使用
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import re
from datetime import datetime, timedelta

# ===================== 配置（需与训练脚本一致）=====================
MODEL_PATH = r"F:\兴隆\epoch\transformer_6class\s1_best_a_precision.pt"
SOURCE_DIR = r"F:\兴隆\数据集"
TARGET_YEAR = "2012"
TARGET_MONTH = "03"
IMG_SIZE = 384
CENTER_CROP_SIZE = 512
EMBED_DIM = 160
NUM_HEADS = 5
DEPTH = 5
CNN_CHANNELS = [32, 64, 128, 160]
NUM_CLASSES = 6
OUTPUT_FILE = r"F:\兴隆\分类结果\transformer_6class\2012_3\inference_results_transformer_6class_2012_03.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'
}
LABEL_FULL_NAMES = {
    0: 'a-星空', 1: 'b-云层', 2: 'c-光球光条',
    3: 'd-曝光', 4: 'e-黑暗', 5: 'f-异常'
}

# ===================== 模型定义 =====================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class EnhancedCNN(nn.Module):
    def __init__(self, channels=CNN_CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            DepthwiseSeparableConv(c1, c2, stride=2),
            DepthwiseSeparableConv(c2, c2),
            DepthwiseSeparableConv(c2, c3, stride=2),
            DepthwiseSeparableConv(c3, c3),
            DepthwiseSeparableConv(c3, c4, stride=2),
            DepthwiseSeparableConv(c4, c4),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return x.flatten(1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=3.0, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class EnhancedTransformerCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, patch_size=6, cnn_channels=CNN_CHANNELS):
        super().__init__()
        self.cnn = EnhancedCNN(channels=cnn_channels)
        num_patches = patch_size * patch_size
        self.patch_embed = nn.Linear(cnn_channels[-1], embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.cnn(x)
        x = x.view(B, 36, -1)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        return self.head(x[:, 0])


class BinaryClassifier(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
                 patch_size=6, cnn_channels=CNN_CHANNELS):
        super().__init__()
        self.cnn = EnhancedCNN(channels=cnn_channels)
        num_patches = patch_size * patch_size
        self.patch_embed = nn.Linear(cnn_channels[-1], embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.cnn(x)
        x = x.view(B, 36, -1)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        return self.head(x[:, 0]).squeeze(-1)


# ===================== 工具函数 =====================
def crop_center(img, crop_size=CENTER_CROP_SIZE):
    width, height = img.size
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def parse_filename(filename):
    """从文件名提取日期和时间"""
    match = re.search(r'CA_(\d{8})_(\d{6})', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_date_folder(date_str, time_str):
    """计算实际观测日期文件夹"""
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    hour = int(time_str[:2])
    if hour < 8:
        dt = datetime(int(year), int(month), int(day)) - timedelta(days=1)
        year, month, day = dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')
    return f"CA_{year}_{month}{day}"


def load_image(path):
    img = Image.open(path)
    img_array = np.array(img).astype(np.float32)
    img_min, img_max = img_array.min(), img_array.max()
    img_array = (img_array - img_min) / (img_max - img_min + 1e-8)
    img_array = (img_array * 255).astype(np.uint8)
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img = crop_center(img)
    return img


# ===================== 图像变换 =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ===================== 单阶段推理 =====================
def predict_single_stage(model, image_tensor, device):
    """使用 epoch35 的 5 类模型进行单阶段推理。"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()

    info = {
        'pred': pred,
        'conf': conf,
        'probs': probs[0].cpu().numpy().tolist(),
    }
    return pred, conf, info


# ===================== 主函数 =====================
def main():
    print(f"使用设备: {device}")

    # ---- 加载模型 ----
    print(f"\n加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先确认六分类 Transformer 模型文件存在")
        return

    bundle = torch.load(MODEL_PATH, weights_only=False, map_location=device)

    model = EnhancedTransformerCNN(num_classes=NUM_CLASSES).to(device)

    if 'stage1_model' in bundle:
        model.load_state_dict(bundle['stage1_model'])
        print(f"  Stage1 a类精确率: {bundle['stage1_a_precision']:.4f}, 召回率: {bundle['stage1_a_recall']:.4f}")
    else:
        model.load_state_dict(bundle['model_state_dict'])
        if 'a_precision' in bundle and 'a_recall' in bundle:
            print(f"  a类精确率: {bundle['a_precision']:.4f}, 召回率: {bundle['a_recall']:.4f}")

    model.eval()
    print("模型加载完成")

    # ---- 遍历图片 ----
    print(f"\n开始推理: {SOURCE_DIR}")

    if not os.path.exists(SOURCE_DIR):
        print(f"[错误] 目录不存在: {SOURCE_DIR}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    processed = 0
    skipped = 0

    # CSV 表头：0=路径, 1=文件名, 2=预测名称, 3=预测标签, 4=置信度, 5=原始文件夹, 后面为各类别概率
    fieldnames = [
        'filepath', 'filename', 'pred_name', 'pred_idx', 'conf',
        'original_folder',
        'prob_a', 'prob_b', 'prob_c', 'prob_d', 'prob_e', 'prob_f'
    ]

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        year_dir = os.path.join(SOURCE_DIR, TARGET_YEAR)
        if not os.path.exists(year_dir):
            print(f"[错误] 年份目录不存在: {year_dir}")
            return

        for folder_name in sorted(os.listdir(year_dir)):
            if not folder_name.startswith(TARGET_MONTH):
                continue

            folder_path = os.path.join(year_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for filename in sorted(os.listdir(folder_path)):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    continue

                image_path = os.path.join(folder_path, filename)
                try:
                    img = load_image(image_path)
                    img_tensor = transform(img)

                    pred, conf, info = predict_single_stage(model, img_tensor, device)

                    probs = info['probs']
                    label_char = LABEL_NAMES[pred]
                    full_name = LABEL_FULL_NAMES[pred]

                    row = {
                        'filepath': image_path,
                        'filename': filename,
                        'pred_name': full_name,
                        'pred_idx': pred,
                        'conf': round(conf, 6),
                        'original_folder': folder_name,
                        'prob_a': round(probs[0], 6),
                        'prob_b': round(probs[1], 6),
                        'prob_c': round(probs[2], 6),
                        'prob_d': round(probs[3], 6),
                        'prob_e': round(probs[4], 6),
                        'prob_f': round(probs[5], 6),
                    }
                    writer.writerow(row)

                    processed += 1
                    if processed % 500 == 0:
                        print(f"  已处理: {processed} 张图片...")

                except Exception as e:
                    skipped += 1
                    continue

    print(f"\n推理完成!")
    print(f"  处理: {processed} 张图片")
    print(f"  跳过: {skipped} 张")
    print(f"  结果保存至: {OUTPUT_FILE}")

    # ---- 统计 ----
    pred_counts = {i: 0 for i in range(len(LABEL_NAMES))}
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['pred_idx'])
                pred_counts[idx] += 1
            except (ValueError, KeyError):
                pass

    print(f"\n预测统计:")
    for idx, count in sorted(pred_counts.items()):
        print(f"  {LABEL_FULL_NAMES[idx]}: {count}")


if __name__ == "__main__":
    main()
