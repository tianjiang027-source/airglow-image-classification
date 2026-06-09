"""
第一阶段 Transformer + CNN 图像分类训练

【改进点】
1. 适度提升模型容量：EMBED_DIM=160, NUM_HEADS=5, DEPTH=5
2. 调整采样权重：e/d类权重上限 4.0，防止过度采样干扰
3. 标准交叉熵损失（加权）：去除 Focal Loss 的不稳定性
4. Label Smoothing (0.1)
5. 更保守的 OneCycleLR：max_lr=5e-4
6. 更保守的数据增强：去除 MixUp（对均衡采样+小模型来说太激进）
7. 单阶段训练：只训练多分类模型
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局配置 =====================
TRAIN_SET_DIR = r"F:\兴隆\训练集\class_train\4(5+异常)"
TRAINING_FILE = r"F:\兴隆\训练集\class_train\4(5+异常)\classification_results.txt"
OUTPUT_DIR = r"F:\兴隆\epoch\transformer_6class"
RECORD_DIR = os.path.join(OUTPUT_DIR, "training_records")

IMG_SIZE = 384
BATCH_SIZE = 24
NUM_EPOCHS_STAGE1 = 40
LEARNING_RATE = 2e-4
NUM_CLASSES = 6
TRAIN_RATIO = 0.8
SEED = 42
CENTER_CROP_RATIO = 0.6

EMBED_DIM = 160
NUM_HEADS = 5
DEPTH = 5
CNN_CHANNELS = [32, 64, 128, 160]

A_CLASS_CONF_THRESHOLD = 0.40
MAX_CLASS_WEIGHT = 4.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"使用设备: {device} | 混合精度训练: {USE_AMP}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)


# ===================== 工具函数 =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def crop_center(img, ratio=CENTER_CROP_RATIO):
    width, height = img.size
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return img.crop((left, top, left + new_width, top + new_height))

def parse_filename(filename):
    if filename.startswith('proj_'):
        filename = filename[5:]
    match = re.match(r'(.+?)_(\d{8})_(\d{6})\.tif', filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

def get_date_folder(date_str, time_str):
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    hour = int(time_str[:2])
    if hour < 8:
        dt = datetime(int(year), int(month), int(day)) - timedelta(days=1)
        year, month, day = dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')
    return f"CA_{year}_{month}{day}"

def load_classification_results(result_txt):
    label_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
    label_names = {
        0: 'a-星空', 1: 'b-云层', 2: 'c-光球光条',
        3: 'd-曝光', 4: 'e-黑暗', 5: 'f-异常'
    }
    samples = []
    with open(result_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[-1] in label_map:
                samples.append((" ".join(parts[:-1]), label_map[parts[-1]]))
    return samples, label_map, label_names


# ===================== 数据集 =====================
class AstroImageDataset(Dataset):
    def __init__(self, train_set_dir, samples, crop_center=True, crop_ratio=CENTER_CROP_RATIO):
        self.train_set_dir = train_set_dir
        self.crop_center = crop_center
        self.crop_ratio = crop_ratio
        self.folder_map = {
            0: 'a_星空', 1: 'b_云层', 2: 'c_光球光条',
            3: 'd_曝光', 4: 'e_黑暗', 5: 'f_异常'
        }
        self.valid_samples = []
        for filename, label in samples:
            folder_name = self.folder_map.get(label)
            if folder_name is None:
                continue
            class_path = os.path.join(train_set_dir, folder_name, os.path.basename(filename))
            direct_path = os.path.join(train_set_dir, filename)
            if os.path.exists(class_path):
                self.valid_samples.append((class_path, label))
            elif os.path.exists(direct_path):
                self.valid_samples.append((direct_path, label))
        print(f"  有效样本: {len(self.valid_samples)} / {len(samples)}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        path, label = self.valid_samples[idx]
        try:
            img = Image.open(path)
            img_array = np.array(img).astype(np.float32)
            img_min, img_max = img_array.min(), img_array.max()
            img_array = (img_array - img_min) / (img_max - img_min + 1e-8)
            img_array = (img_array * 255).astype(np.uint8)
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            if self.crop_center:
                img = crop_center(img, self.crop_ratio)
            return img, label, path
        except:
            return Image.new('L', (IMG_SIZE, IMG_SIZE)), label, path

    def get_class_distribution(self):
        counter = [0] * NUM_CLASSES
        for _, label in self.valid_samples:
            counter[label] += 1
        return counter


class BinaryAstroDataset(Dataset):
    def __init__(self, samples_a, samples_cf, crop_center=True, crop_ratio=CENTER_CROP_RATIO):
        self.crop_center = crop_center
        self.crop_ratio = crop_ratio
        self.samples = []
        for path, _ in samples_a:
            self.samples.append((path, 0))
        for path, _ in samples_cf:
            self.samples.append((path, 1))
        np.random.seed(SEED)
        np.random.shuffle(self.samples)
        print(f"  二分类样本: a类={len(samples_a)}, c/f类={len(samples_cf)}, 总计={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path)
            img_array = np.array(img).astype(np.float32)
            img_min, img_max = img_array.min(), img_array.max()
            img_array = (img_array - img_min) / (img_max - img_min + 1e-8)
            img_array = (img_array * 255).astype(np.uint8)
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            if self.crop_center:
                img = crop_center(img, self.crop_ratio)
            return img, label, path
        except:
            return Image.new('L', (IMG_SIZE, IMG_SIZE)), label, path


class TransformDataset(Dataset):
    def __init__(self, subset, train_mode=False):
        self.subset = subset
        self.train_mode = train_mode
        if self.train_mode:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label, path = self.subset[idx]
        img = self.transform(img)
        return img, label


# ===================== 损失函数 =====================
class WeightedLabelSmoothingLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.1, num_classes=6):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        loss = -smooth_targets * log_probs
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float32, device=inputs.device)
            loss = weights.unsqueeze(0) * loss
        return loss.sum(dim=1).mean()


class BinaryLabelSmoothingBCELoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        targets_float = targets.float()
        if self.label_smoothing > 0:
            targets_float = targets_float * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets_float)


# ===================== 模型 =====================
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
    def __init__(self, dim, num_heads=5, mlp_ratio=3.0, dropout=0.1):
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
    def __init__(self, num_classes=6, embed_dim=EMBED_DIM, depth=DEPTH,
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
            nn.Dropout(0.15),
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


# ===================== 训练函数 =====================
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, num_classes=6):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_predicted = [0] * num_classes
    class_true_positive = [0] * num_classes

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            class_predicted[pred] += 1
            if pred == label:
                class_correct[label] += 1
                class_true_positive[label] += 1

        if (batch_idx + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    class_recall = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    class_precision = [class_true_positive[i] / class_predicted[i] if class_predicted[i] > 0 else 0 for i in range(num_classes)]

    return epoch_loss, epoch_acc, class_recall, class_precision


def train_epoch_binary(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            l, p = labels[i].item(), predicted[i].item()
            if l == 0 and p == 0:
                tp += 1
            elif l != 0 and p == 0:
                fp += 1
            elif l == 0 and p != 0:
                fn += 1

        if (batch_idx + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return running_loss / len(loader), correct / total, recall, precision


def validate_epoch(model, loader, criterion, device, num_classes=6, is_binary=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_predicted = [0] * num_classes
    class_true_positive = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_binary:
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            else:
                _, predicted = outputs.max(1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                class_predicted[pred] += 1
                if pred == label:
                    class_correct[label] += 1
                    class_true_positive[label] += 1

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    class_recall = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    class_precision = [class_true_positive[i] / class_predicted[i] if class_predicted[i] > 0 else 0 for i in range(num_classes)]

    return epoch_loss, epoch_acc, class_recall, class_precision


def validate_with_threshold(model, loader, device, num_classes=6):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_predicted = [0] * num_classes
    class_true_positive = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            a_probs = probs[:, 0]
            mask = (predicted == 0) & (a_probs < A_CLASS_CONF_THRESHOLD)
            if mask.any():
                probs = probs.clone()
                probs[mask, 0] = 0
                predicted = probs.argmax(dim=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                class_predicted[pred] += 1
                if pred == label:
                    class_correct[label] += 1
                    class_true_positive[label] += 1

    epoch_acc = correct / total
    class_recall = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    class_precision = [class_true_positive[i] / class_predicted[i] if class_predicted[i] > 0 else 0 for i in range(num_classes)]

    return epoch_acc, class_recall, class_precision


# ===================== 可视化 =====================
def plot_training_curves(record_file, output_dir, label_names, num_classes=6):
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    class_recall_history = {i: [] for i in range(num_classes)}
    class_precision_history = {i: [] for i in range(num_classes)}

    with open(record_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_loss.append(float(row['Train_Loss']))
            val_loss.append(float(row['Val_Loss']))
            train_acc.append(float(row['Train_Acc']))
            val_acc.append(float(row['Val_Acc']))
            for i in range(num_classes):
                class_recall_history[i].append(float(row[f'Class_{i}_Recall']))
                class_precision_history[i].append(float(row[f'Class_{i}_Precision']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_curves.png'), dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

    for i in range(num_classes):
        axes[i].plot(epochs, class_recall_history[i], color=colors[i], linewidth=2, label='Recall')
        axes[i].plot(epochs, class_precision_history[i], color=colors[i], linewidth=2, linestyle='--', label='Precision')
        axes[i].set_title(label_names[i], fontsize=10, fontweight='bold' if i == 0 else 'normal')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Rate')
        axes[i].set_ylim([0, 1.05])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='lower right')
        max_idx = np.argmax(class_precision_history[i])
        max_val = class_precision_history[i][max_idx]
        axes[i].annotate(f'Pmax: {max_val:.2%}', xy=(epochs[max_idx], max_val), fontsize=9, color=colors[i])

    for j in range(num_classes, 8):
        axes[j].axis('off')
    axes[0].set_facecolor('#e8f5e9')
    axes[0].set_title('★ a-星空(适合观测) [重点]', fontsize=10, fontweight='bold', color='#27ae60')
    plt.suptitle('Recall vs Precision by Class (Recall实线, Precision虚线)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy_curves.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, class_precision_history[0], color='#e74c3c', linewidth=2.5, marker='o', markersize=4, label='Precision (精确率)')
    ax.plot(epochs, class_recall_history[0], color='#3498db', linewidth=2.5, marker='s', markersize=4, linestyle='--', label='Recall (召回率)')
    ax.fill_between(epochs, 0, class_precision_history[0], alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('a-星空 - 精确率 vs 召回率\n[第一阶段模型]', fontsize=12, fontweight='bold', color='#27ae60')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='right')
    max_idx = np.argmax(class_precision_history[0])
    max_val = class_precision_history[0][max_idx]
    ax.annotate(f'最佳精确率: {max_val:.2%} (Epoch {epochs[max_idx]})',
               xy=(epochs[max_idx], max_val),
               xytext=(epochs[max_idx] + 3, max_val - 0.15),
               fontsize=11, color='#e74c3c', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_a_accuracy.png'), dpi=150)
    plt.close()
    print("可视化图表已保存")


# ===================== 主函数 =====================
def main():
    set_seed(SEED)

    print("=" * 60)
    print("第一阶段 Transformer + CNN 图像分类训练")
    print("=" * 60)

    print(f"\n1. 加载训练数据...")
    samples, label_map, label_names = load_classification_results(TRAINING_FILE)
    print(f"   总样本数: {len(samples)}")

    print(f"\n2. 准备数据集...")
    full_dataset = AstroImageDataset(TRAIN_SET_DIR, samples, crop_center=True)
    print(f"   有效样本: {len(full_dataset)}")

    train_size = int(len(full_dataset) * TRAIN_RATIO)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    class_dist = full_dataset.get_class_distribution()
    print(f"   类别分布: {class_dist}")
    print(f"   训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    total_samples = sum(class_dist)
    class_weights = []
    for c in class_dist:
        w = total_samples / (NUM_CLASSES * c) if c > 0 else 1.0
        class_weights.append(min(w, MAX_CLASS_WEIGHT))
    print(f"   均衡权重(上限{MAX_CLASS_WEIGHT}): {[f'{w:.2f}' for w in class_weights]}")

    train_labels = []
    for idx in train_dataset.indices:
        _, label = full_dataset.valid_samples[idx]
        train_labels.append(label)

    class_counts = [train_labels.count(i) for i in range(NUM_CLASSES)]
    sample_weights = []
    for label in train_labels:
        w = total_samples / (NUM_CLASSES * class_counts[label]) if class_counts[label] > 0 else 1.0
        sample_weights.append(min(w, MAX_CLASS_WEIGHT))

    balanced_sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
    print(f"   均衡采样: 每个epoch各类别被采样 ~{len(train_labels) // NUM_CLASSES} 次")

    train_dataset_t = TransformDataset(train_dataset, train_mode=True)
    val_dataset_t = TransformDataset(val_dataset, train_mode=False)

    train_loader = DataLoader(train_dataset_t, batch_size=BATCH_SIZE, sampler=balanced_sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset_t, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = os.path.join(RECORD_DIR, f"training_record_{timestamp}.csv")
    fieldnames = ['Epoch', 'Stage', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc']
    fieldnames += [f'Class_{i}_Recall' for i in range(NUM_CLASSES)]
    fieldnames += [f'Class_{i}_Precision' for i in range(NUM_CLASSES)]
    fieldnames += ['Learning_Rate', 'Model_Path']

    with open(record_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # =========================================================================
    # 第一阶段
    # =========================================================================
    print(f"\n{'='*60}")
    print("【第一阶段】5类分类器")
    print(f"{'='*60}")

    model_s1 = EnhancedTransformerCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model_s1.parameters())
    print(f"   总参数量: {total_params:,} (~{total_params * 4 / 1024 / 1024:.1f} MB)")

    criterion_s1 = WeightedLabelSmoothingLoss(
        class_weights=class_weights, label_smoothing=0.1, num_classes=NUM_CLASSES
    )
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS_STAGE1
    scheduler_s1 = optim.lr_scheduler.OneCycleLR(
        optimizer_s1, max_lr=LEARNING_RATE * 5,
        total_steps=total_steps, pct_start=0.15,
        anneal_strategy='cos', div_factor=25, final_div_factor=1000
    )
    scaler_s1 = torch.amp.GradScaler('cuda') if USE_AMP else None

    print(f"\n3. 开始第一阶段训练 ({NUM_EPOCHS_STAGE1} epochs)...")
    print(f"   max_lr: {LEARNING_RATE * 5:.1e}, label_smoothing: 0.1")

    best_s1_acc = 0.0
    best_s1_a_precision = -1.0
    best_s1_path = os.path.join(OUTPUT_DIR, "s1_best_a_precision.pt")

    for epoch in range(1, NUM_EPOCHS_STAGE1 + 1):
        print(f"\n第一阶段 Epoch {epoch}/{NUM_EPOCHS_STAGE1}")
        print("-" * 40)

        train_loss, train_acc, train_recall, train_precision = train_epoch(
            model_s1, train_loader, criterion_s1, optimizer_s1, scheduler_s1, scaler_s1, device, num_classes=NUM_CLASSES)

        current_lr = optimizer_s1.param_groups[0]['lr']

        val_loss, val_acc, val_recall, val_precision = validate_epoch(
            model_s1, val_loader, criterion_s1, device, num_classes=NUM_CLASSES)

        val_acc_th, val_recall_th, val_precision_th = validate_with_threshold(
            model_s1, val_loader, device, num_classes=NUM_CLASSES)

        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"  ★ a-星空 召回率: {val_recall[0]:.4f} | 精确率: {val_precision[0]:.4f}")
        print(f"  ★ f-异常 召回率: {val_recall[5]:.4f} | 精确率: {val_precision[5]:.4f}")
        print(f"    (阈值后: 召回率: {val_recall_th[0]:.4f} | 精确率: {val_precision_th[0]:.4f})")

        model_name = f"s1_epoch_{epoch:02d}.pt"
        model_path = os.path.join(OUTPUT_DIR, model_name)
        torch.save({
            'epoch': epoch, 'stage': 1,
            'model_state_dict': model_s1.state_dict(),
            'val_acc': val_acc, 'a_precision': val_precision[0],
        }, model_path)
        print(f"  已保存: {model_name}")

        with open(record_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            row = {
                'Epoch': epoch, 'Stage': 1,
                'Train_Loss': f"{train_loss:.6f}", 'Train_Acc': f"{train_acc:.6f}",
                'Val_Loss': f"{val_loss:.6f}", 'Val_Acc': f"{val_acc:.6f}",
                'Learning_Rate': f"{current_lr:.8f}", 'Model_Path': model_path
            }
            for i in range(NUM_CLASSES):
                row[f'Class_{i}_Recall'] = f"{val_recall[i]:.6f}"
                row[f'Class_{i}_Precision'] = f"{val_precision[i]:.6f}"
            writer.writerow(row)

        if val_acc > best_s1_acc:
            best_s1_acc = val_acc
        if val_precision_th[0] > best_s1_a_precision:
            best_s1_a_precision = val_precision_th[0]
            torch.save({'epoch': epoch, 'model_state_dict': model_s1.state_dict(),
                        'a_precision': val_precision_th[0], 'a_recall': val_recall_th[0],
                        'f_precision': val_precision_th[5], 'f_recall': val_recall_th[5],
                        'val_acc': val_acc_th}, best_s1_path)
            print(f"  ★ 新最佳a类精确率: {val_precision_th[0]:.4f}")

    print(f"\n第一阶段完成 | 最佳验证准确率: {best_s1_acc:.4f} | 最佳a类精确率: {best_s1_a_precision:.4f}")

    # =========================================================================
    # 保存最终模型
    # =========================================================================
    print(f"\n{'='*60}")
    print("【保存最终模型】")
    print(f"{'='*60}")

    best_s1_state = torch.load(best_s1_path, weights_only=False, map_location=device)

    final_bundle = {
        'model_state_dict': best_s1_state['model_state_dict'],
        'stage1_a_precision': best_s1_state['a_precision'],
        'stage1_a_recall': best_s1_state['a_recall'],
        'config': {
            'embed_dim': EMBED_DIM, 'num_heads': NUM_HEADS, 'depth': DEPTH,
            'cnn_channels': CNN_CHANNELS, 'img_size': IMG_SIZE,
            'num_classes': NUM_CLASSES, 'center_crop_ratio': CENTER_CROP_RATIO,
            'a_class_conf_threshold': A_CLASS_CONF_THRESHOLD
        }
    }
    final_path = os.path.join(OUTPUT_DIR, "stage1_final.pt")
    torch.save(final_bundle, final_path)
    print(f"第一阶段模型已保存: {final_path}")
    print(f"  Stage1: P={best_s1_state['a_precision']:.4f}, R={best_s1_state['a_recall']:.4f}")

    print(f"\n4. 生成可视化图表...")
    plot_training_curves(record_file, RECORD_DIR, label_names, num_classes=NUM_CLASSES)

    print(f"\n{'='*60}")
    print("第一阶段训练完成!")
    print(f"Stage1 最佳a类精确率: {best_s1_a_precision:.4f}")
    print(f"所有文件保存在: {OUTPUT_DIR}")
    print(f"训练记录: {record_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
