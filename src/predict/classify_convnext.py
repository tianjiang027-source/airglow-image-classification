"""
使用 ConvNeXt-Tiny 星点特征增强模型进行分类。

必须与 convnext_star_features训练.py 使用相同预处理：
  R: 固定范围归一化后的绝对亮度
  G: 百分位拉伸后的局部可见结构
  B: 星点候选/高频亮点结构
"""

import csv
import os
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import convnext_tiny


# ===================== 配置 =====================
MODEL_PATH = r"F:\兴隆\epoch\convnext_star_features_3_6class\convnext_star_features_best.pt"
SOURCE_DIR = r"F:\兴隆\数据集"
OUTPUT_FILE = r"F:\兴隆\分类结果\convnext_star_features_3_6class\2012_all\inference_results_star_features_2012_all.csv"

TARGET_YEAR = "2012"
TARGET_MONTH = None  # None 表示跑全年；例如改成 "03" 就只跑 3 月
IMG_SIZE = 224
CENTER_CROP_SIZE = 512
NUM_CLASSES = 6
FIXED_NORMALIZE_MAX = 65535.0
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.7
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

LABEL_CHARS = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
}

LABEL_FULL_NAMES = {
    0: "a-星空",
    1: "b-云层",
    2: "c-光球光条",
    3: "d-曝光",
    4: "e-黑暗",
    5: "f-异常",
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model():
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    return model


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    bundle = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    print(f"已加载模型: {model_path}")
    if "epoch" in bundle:
        print(f"模型 epoch: {bundle['epoch']}")
    if "val_acc" in bundle:
        print(f"验证准确率: {bundle['val_acc']:.4f}")
    if "a_precision" in bundle and "a_recall" in bundle:
        print(f"a-星空 P/R: {bundle['a_precision']:.4f} / {bundle['a_recall']:.4f}")
    if "f_precision" in bundle and "f_recall" in bundle:
        print(f"f-异常 P/R: {bundle['f_precision']:.4f} / {bundle['f_recall']:.4f}")
    if "preprocess" in bundle:
        print(f"预处理: {bundle['preprocess'].get('type', 'unknown')}")

    return model


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
        arr = arr.astype(np.float32).mean(axis=2)
    else:
        arr = arr.astype(np.float32)

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

    detail_threshold = max(float(np.percentile(detail, 97.5)), float(detail.mean() + 2.5 * detail.std()))
    contrast_threshold = max(float(np.percentile(contrast, 98.5)), float(contrast.mean() + 2.0 * contrast.std()))
    star_candidates = ((detail >= detail_threshold) & (contrast >= contrast_threshold)).astype(np.float32)

    rgb = np.stack([absolute, contrast, star_candidates], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def image_belongs_to_target_period(path):
    filename = os.path.basename(path)

    match = re.match(r"^(20\d{2})(\d{2})(\d{2})", filename)
    if match:
        if match.group(1) != TARGET_YEAR:
            return False
        return TARGET_MONTH is None or match.group(2) == TARGET_MONTH

    if TARGET_MONTH is None:
        return True
    parent = os.path.basename(os.path.dirname(path))
    return parent.startswith(TARGET_MONTH)


def iter_target_images(source_dir):
    year_dir = os.path.join(source_dir, TARGET_YEAR)
    scan_root = year_dir if os.path.isdir(year_dir) else source_dir

    for root, dirs, files in os.walk(scan_root):
        dirs.sort()
        files.sort()
        for filename in files:
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(root, filename)
            if image_belongs_to_target_period(path):
                yield path


def predict(model, image_path):
    img = make_star_feature_image(image_path)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        conf = float(probs[pred_idx].item())

    return pred_idx, conf, probs.cpu().numpy().tolist()


def main():
    print("=" * 60)
    print("ConvNeXt-Tiny 星点特征增强模型分类")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"模型文件: {MODEL_PATH}")
    print(f"数据目录: {SOURCE_DIR}")
    print(f"目标年份: {TARGET_YEAR}")
    target_month_text = "全年" if TARGET_MONTH is None else TARGET_MONTH
    print(f"目标月份: {target_month_text}")
    print(f"输出文件: {OUTPUT_FILE}")

    model = load_model(MODEL_PATH)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    image_paths = list(iter_target_images(SOURCE_DIR))
    print(f"\n找到目标图片: {len(image_paths)} 张")

    fieldnames = [
        "filepath",
        "filename",
        "pred_name",
        "pred_label",
        "pred_idx",
        "conf",
        "year",
        "original_folder",
        "relative_folder",
        "prob_a",
        "prob_b",
        "prob_c",
        "prob_d",
        "prob_e",
        "prob_f",
    ]

    counts = {i: 0 for i in range(NUM_CLASSES)}
    processed = 0
    failed = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in image_paths:
            try:
                pred_idx, conf, probs = predict(model, image_path)
            except Exception as exc:
                print(f"[失败] {image_path}: {exc}")
                failed += 1
                continue

            rel_folder = os.path.relpath(os.path.dirname(image_path), SOURCE_DIR)
            parts = rel_folder.split(os.sep)
            year = parts[0] if parts else ""
            original_folder = parts[-1] if parts else ""

            writer.writerow({
                "filepath": image_path,
                "filename": os.path.basename(image_path),
                "pred_name": LABEL_FULL_NAMES[pred_idx],
                "pred_label": LABEL_CHARS[pred_idx],
                "pred_idx": pred_idx,
                "conf": round(conf, 6),
                "year": year,
                "original_folder": original_folder,
                "relative_folder": rel_folder,
                "prob_a": round(probs[0], 6),
                "prob_b": round(probs[1], 6),
                "prob_c": round(probs[2], 6),
                "prob_d": round(probs[3], 6),
                "prob_e": round(probs[4], 6),
                "prob_f": round(probs[5], 6),
            })

            counts[pred_idx] += 1
            processed += 1

            if processed % 500 == 0:
                print(f"已处理 {processed} / {len(image_paths)}")

    print("\n分类完成")
    print(f"成功处理: {processed}")
    print(f"失败: {failed}")
    print(f"结果保存至: {OUTPUT_FILE}")
    print("\n预测统计:")
    for idx in range(NUM_CLASSES):
        print(f"  {LABEL_FULL_NAMES[idx]}: {counts[idx]}")


if __name__ == "__main__":
    main()
