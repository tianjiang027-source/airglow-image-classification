"""
ConvNeXt-Tiny 预训练微调：星点特征增强版

核心思路：
1. 使用固定范围保留绝对亮度，避免白图变黑图。
2. 增加局部高频细节通道，让模型更容易区分星点和云层。
3. 增加星点候选通道，突出小而尖锐的亮点结构。

输入三通道含义：
  R: 固定范围归一化后的绝对亮度
  G: 百分位拉伸后的局部可见结构
  B: 星点候选/高频亮点结构
"""

import csv
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


# ===================== 配置 =====================
TRAIN_SET_DIR = r"F:\兴隆\训练集\class_train\4(5+异常)"
TRAINING_FILE = r"F:\兴隆\训练集\class_train\4(5+异常)\classification_results.txt"
OUTPUT_DIR = r"F:\兴隆\epoch\convnext_star_features_3_6class"
RECORD_DIR = os.path.join(OUTPUT_DIR, "training_records")

NUM_CLASSES = 6
IMG_SIZE = 224
CENTER_CROP_SIZE = 512
BATCH_SIZE = 24
NUM_EPOCHS = 40
FREEZE_EPOCHS = 5
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MAX_CLASS_WEIGHT = 4.0

LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

FIXED_NORMALIZE_MAX = 65535.0
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.7

LABEL_MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
LABEL_NAMES = {
    0: "a-星空",
    1: "b-云层",
    2: "c-光球光条",
    3: "d-曝光",
    4: "e-黑暗",
    5: "f-异常",
}
FOLDER_MAP = {
    0: "a_星空",
    1: "b_云层",
    2: "c_光球光条",
    3: "d_曝光",
    4: "e_黑暗",
    5: "f_异常",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ===================== 初始化 =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"使用设备: {device} | 混合精度训练: {use_amp}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ===================== 数据 =====================
def load_classification_results(result_txt):
    samples = []
    with open(result_txt, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[跳过] 第 {line_no} 行格式不正确: {line}")
                continue

            filename = " ".join(parts[:-1])
            label_char = parts[-1].lower()
            if label_char not in LABEL_MAP:
                print(f"[跳过] 第 {line_no} 行未知类别: {line}")
                continue

            samples.append((filename, LABEL_MAP[label_char]))

    return samples


def crop_center_array(arr, crop_size):
    height, width = arr.shape[:2]
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return arr[top:top + crop_h, left:left + crop_w]


def image_to_fixed_float(path):
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3:
        arr = arr.astype(np.float32)
        arr = arr.mean(axis=2)
    else:
        arr = arr.astype(np.float32)

    # PIL 的 I 模式有时会变成 int32，但原始数据仍按 16 位天文图处理。
    max_value = 255.0 if float(arr.max()) <= 255.0 else FIXED_NORMALIZE_MAX
    arr = np.clip(arr, 0, max_value) / max_value
    return crop_center_array(arr, CENTER_CROP_SIZE)


def percentile_stretch(arr, low=PERCENTILE_LOW, high=PERCENTILE_HIGH):
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    stretched = (arr - lo) / (hi - lo)
    return np.clip(stretched, 0, 1).astype(np.float32)


def normalize_feature(arr):
    arr = arr.astype(np.float32)
    hi = float(np.percentile(arr, 99.5))
    if hi < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / hi, 0, 1).astype(np.float32)


def make_star_feature_image(path):
    absolute = image_to_fixed_float(path)
    contrast = percentile_stretch(absolute)

    contrast_img = Image.fromarray((contrast * 255).astype(np.uint8), mode="L")
    blur = np.array(contrast_img.filter(ImageFilter.GaussianBlur(radius=2.0))).astype(np.float32) / 255.0
    detail = np.clip(contrast - blur, 0, 1)
    detail = normalize_feature(detail)

    # 小而尖锐的亮点更像星点；云层通常是低频连续结构。
    detail_threshold = max(float(np.percentile(detail, 97.5)), float(detail.mean() + 2.5 * detail.std()))
    contrast_threshold = max(float(np.percentile(contrast, 98.5)), float(contrast.mean() + 2.0 * contrast.std()))
    star_candidates = ((detail >= detail_threshold) & (contrast >= contrast_threshold)).astype(np.float32)

    rgb = np.stack([absolute, contrast, star_candidates], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


class AstroClassDataset(Dataset):
    def __init__(self, train_set_dir, samples):
        self.train_set_dir = train_set_dir
        self.valid_samples = []

        for filename, label in samples:
            basename = os.path.basename(filename)
            class_dir = FOLDER_MAP[label]
            class_path = os.path.join(train_set_dir, class_dir, basename)
            direct_path = os.path.join(train_set_dir, basename)

            if os.path.exists(class_path):
                self.valid_samples.append((class_path, label))
            elif os.path.exists(direct_path):
                self.valid_samples.append((direct_path, label))

        print(f"  有效样本: {len(self.valid_samples)} / {len(samples)}")

    def __len__(self):
        return len(self.valid_samples)

    def get_class_distribution(self):
        counter = [0] * NUM_CLASSES
        for _, label in self.valid_samples:
            counter[label] += 1
        return counter


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class TransformSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base_dataset.valid_samples[self.indices[idx]]
        try:
            img = make_star_feature_image(path)
        except Exception as exc:
            print(f"[读图失败] {path}: {exc}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        return self.transform(img), label


# ===================== 模型 =====================
def build_model():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("classifier.2"):
            param.requires_grad = trainable


def make_optimizer(model, lr):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def compute_class_metrics(targets, preds):
    recall = []
    precision = []
    for cls in range(NUM_CLASSES):
        tp = sum(1 for t, p in zip(targets, preds) if t == cls and p == cls)
        fn = sum(1 for t, p in zip(targets, preds) if t == cls and p != cls)
        fp = sum(1 for t, p in zip(targets, preds) if t != cls and p == cls)
        recall.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        precision.append(tp / (tp + fp) if tp + fp > 0 else 0.0)
    return recall, precision


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    recall, precision = compute_class_metrics(all_targets, all_preds)
    return total_loss / len(loader), correct / total, recall, precision


def save_checkpoint(path, model, epoch, val_acc, a_precision, a_recall, f_precision, f_recall):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "a_precision": a_precision,
        "a_recall": a_recall,
        "f_precision": f_precision,
        "f_recall": f_recall,
        "label_names": LABEL_NAMES,
        "preprocess": {
            "type": "absolute_contrast_star_candidates",
            "fixed_normalize_max": FIXED_NORMALIZE_MAX,
            "center_crop_size": CENTER_CROP_SIZE,
            "img_size": IMG_SIZE,
            "channels": ["absolute_brightness", "percentile_contrast", "star_candidates"],
        },
    }, path)


def main():
    print("=" * 60)
    print("ConvNeXt-Tiny 星点特征增强训练")
    print("=" * 60)
    print(f"训练集: {TRAIN_SET_DIR}")
    print(f"标注文件: {TRAINING_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")

    samples = load_classification_results(TRAINING_FILE)
    print(f"\n1. 加载训练数据: {len(samples)} 条")

    base_dataset = AstroClassDataset(TRAIN_SET_DIR, samples)
    class_dist = base_dataset.get_class_distribution()
    print(f"   类别分布: {class_dist}")
    for i, count in enumerate(class_dist):
        print(f"   {LABEL_NAMES[i]}: {count}")

    train_size = int(len(base_dataset) * TRAIN_RATIO)
    val_size = len(base_dataset) - train_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    train_indices = train_subset.indices
    val_indices = val_subset.indices
    train_dataset = TransformSubset(base_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(base_dataset, val_indices, val_transform)

    train_labels = [base_dataset.valid_samples[idx][1] for idx in train_indices]
    train_counts = Counter(train_labels)
    sample_weights = []
    for label in train_labels:
        weight = len(train_labels) / (NUM_CLASSES * train_counts[label])
        sample_weights.append(min(weight, MAX_CLASS_WEIGHT))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model().to(device)
    set_backbone_trainable(model, False)
    optimizer = make_optimizer(model, LR_HEAD)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    best_path = os.path.join(OUTPUT_DIR, "convnext_star_features_best.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = os.path.join(RECORD_DIR, f"convnext_star_features_record_{timestamp}.csv")

    with open(record_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc",
            "a_precision", "a_recall",
            "b_precision", "b_recall",
            "f_precision", "f_recall",
            "score"
        ])

        print("\n2. 开始训练")
        phase = "freeze_backbone"
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            print("-" * 40)

            if epoch == 1:
                print("  阶段: 冻结主干，只训练分类头")
            if epoch == FREEZE_EPOCHS + 1:
                print("  阶段: 解冻全模型微调")
                set_backbone_trainable(model, True)
                optimizer = make_optimizer(model, LR_FINETUNE)
                phase = "finetune_all"

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_acc, val_recall, val_precision = validate(model, val_loader, criterion)

            a_precision = val_precision[0]
            a_recall = val_recall[0]
            b_precision = val_precision[1]
            b_recall = val_recall[1]
            f_precision = val_precision[5]
            f_recall = val_recall[5]

            # 重点目标：星空不能乱判，也不能漏判；异常图要能被剔除。
            score = (
                0.35 * a_precision
                + 0.35 * a_recall
                + 0.10 * val_acc
                + 0.10 * b_recall
                + 0.10 * f_recall
            )

            print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            print(f"  ★ a-星空 精确率: {a_precision:.4f} | 召回率: {a_recall:.4f}")
            print(f"  ★ b-云层 精确率: {b_precision:.4f} | 召回率: {b_recall:.4f}")
            print(f"  ★ f-异常 精确率: {f_precision:.4f} | 召回率: {f_recall:.4f}")

            epoch_path = os.path.join(OUTPUT_DIR, f"convnext_star_features_epoch_{epoch:02d}.pt")
            save_checkpoint(epoch_path, model, epoch, val_acc, a_precision, a_recall, f_precision, f_recall)
            print(f"  已保存: {epoch_path}")

            if score > best_score:
                best_score = score
                save_checkpoint(best_path, model, epoch, val_acc, a_precision, a_recall, f_precision, f_recall)
                print(f"  ★ 新最佳模型: score={score:.4f}")

            writer.writerow([
                epoch, phase, train_loss, train_acc, val_loss, val_acc,
                a_precision, a_recall, b_precision, b_recall, f_precision, f_recall, score
            ])
            f.flush()

    print("\n" + "=" * 60)
    print("ConvNeXt-Tiny 星点特征增强训练完成")
    print(f"最佳模型: {best_path}")
    print(f"训练记录: {record_file}")
    print(f"所有文件保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
