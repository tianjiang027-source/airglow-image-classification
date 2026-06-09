"""
使用训练好的 ConvNeXt-Tiny 模型分类 F:\兴隆\数据集 中的 1 月图片。

输出:
  F:\兴隆\分类结果\convnext_1\inference_results_convnext_january.csv
"""

import csv
import os
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny


# ===================== 配置 =====================
MODEL_PATH = r"F:\兴隆\epoch\convnext_2\convnext_best.pt"
SOURCE_DIR = r"F:\兴隆\数据集"
OUTPUT_FILE = r"F:\兴隆\分类结果\convnext_2\inference_results_convnext_best_2012_01.csv"

TARGET_MONTH = "01"
TARGET_YEAR = "2012"
IMG_SIZE = 224
CENTER_CROP_SIZE = 512
NUM_CLASSES = 5
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

LABEL_CHARS = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
}

LABEL_FULL_NAMES = {
    0: "a-星空",
    1: "b-云层",
    2: "c-光球光条",
    3: "d-曝光",
    4: "e-黑暗",
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

    return model


def normalize_high_bit_image(img):
    arr = np.array(img)
    if arr.ndim == 3:
        return Image.fromarray(arr).convert("RGB")

    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    arr = (arr - arr_min) / (arr_max - arr_min + 1e-10)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def open_image_as_rgb(path):
    img = Image.open(path)
    if img.mode in ("I", "I;16", "I;16L", "I;16B", "F"):
        img = normalize_high_bit_image(img)
    else:
        img = img.convert("RGB")
    return crop_center(img, CENTER_CROP_SIZE)


def crop_center(img, crop_size):
    width, height = img.size
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def image_belongs_to_target_month(path):
    filename = os.path.basename(path)

    # 优先按文件名开头的 YYYYMMDD 判断，例如 20120101173005...
    match = re.match(r"^(20\d{2})(\d{2})(\d{2})", filename)
    if match:
        return match.group(2) == TARGET_MONTH

    # 兜底按父文件夹判断，例如 0101、0102。
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
            if image_belongs_to_target_month(path):
                yield path


def predict(model, image_path):
    img = open_image_as_rgb(image_path)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        conf = float(probs[pred_idx].item())

    return pred_idx, conf, probs.cpu().numpy().tolist()


def main():
    print("=" * 60)
    print("ConvNeXt-Tiny 一月数据分类")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"数据目录: {SOURCE_DIR}")
    print(f"目标年份: {TARGET_YEAR}")
    print(f"目标月份: {TARGET_MONTH}")
    print(f"输出文件: {OUTPUT_FILE}")

    model = load_model(MODEL_PATH)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    image_paths = list(iter_target_images(SOURCE_DIR))
    print(f"\n找到 1 月图片: {len(image_paths)} 张")

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
            })

            counts[pred_idx] += 1
            processed += 1

            if processed % 500 == 0:
                print(f"已处理: {processed} / {len(image_paths)}")

    print("\n分类完成")
    print(f"成功处理: {processed}")
    print(f"失败: {failed}")
    print(f"结果保存至: {OUTPUT_FILE}")
    print("\n预测统计:")
    for idx in range(NUM_CLASSES):
        print(f"  {LABEL_FULL_NAMES[idx]}: {counts[idx]}")


if __name__ == "__main__":
    main()
