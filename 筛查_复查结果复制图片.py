"""
根据 classification_results.txt 将图片复制到训练集类别子文件夹。

输入:
  F:\兴隆\训练集\class_train\3\classification_results.txt

输出:
  F:\兴隆\训练集\class_train\3\a_星空
  F:\兴隆\训练集\class_train\3\b_云层
  F:\兴隆\训练集\class_train\3\c_光球光条
  F:\兴隆\训练集\class_train\3\d_曝光
  F:\兴隆\训练集\class_train\3\e_黑暗
  F:\兴隆\训练集\class_train\3\f_异常
"""

import os
import shutil
from collections import Counter


RESULT_FILE = r"F:\兴隆\训练集\class_train\4(5+异常)\classification_results.txt"
OUTPUT_DIR = r"F:\兴隆\训练集\class_train\4(5+异常)"

# 按顺序查找源图。先查可能已经整理过的目录，再查原始数据集。
SOURCE_DIRS = [
    r"F:\兴隆\训练集\图片复查",
    r"F:\兴隆\训练集\class_train\2",
    r"F:\兴隆\训练集\class_train\1",
    r"F:\兴隆\数据集",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

LABEL_FOLDERS = {
    "a": "a_星空",
    "b": "b_云层",
    "c": "c_光球光条",
    "d": "d_曝光",
    "e": "e_黑暗",
    "f": "f_异常",
}


def load_annotations(result_file):
    annotations = []
    with open(result_file, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[跳过] 第 {line_no} 行格式不正确: {line}")
                continue

            filename = parts[0]
            label = parts[1].lower()
            if label not in LABEL_FOLDERS:
                print(f"[跳过] 第 {line_no} 行未知类别 {label}: {line}")
                continue

            annotations.append((filename, label))

    return annotations


def build_source_index(source_dirs, wanted_basenames):
    wanted_basenames = set(wanted_basenames)
    source_index = {}

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"[警告] 源目录不存在: {source_dir}")
            continue

        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [
                d for d in dirs
                if d not in {"_review_batches", "deleted_images"}
            ]
            dirs.sort()
            files.sort()

            for filename in files:
                if filename not in wanted_basenames:
                    continue
                if not filename.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                source_index.setdefault(filename, os.path.join(root, filename))

    return source_index


def get_unique_dest_path(dest_dir, filename, src_path):
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        return dest_path, False

    if os.path.getsize(src_path) == os.path.getsize(dest_path):
        return dest_path, True

    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_dest = os.path.join(dest_dir, f"{name}_{counter}{ext}")
        if not os.path.exists(new_dest):
            return new_dest, False
        if os.path.getsize(src_path) == os.path.getsize(new_dest):
            return new_dest, True
        counter += 1


def copy_images():
    if not os.path.exists(RESULT_FILE):
        print(f"[错误] 标注文件不存在: {RESULT_FILE}")
        return

    annotations = load_annotations(RESULT_FILE)
    if not annotations:
        print("未读取到有效标注。")
        return

    print("=" * 60)
    print("根据 classification_results.txt 复制训练图片")
    print("=" * 60)
    print(f"标注文件: {RESULT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"有效标注: {len(annotations)} 条")

    for folder in LABEL_FOLDERS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

    wanted_basenames = [os.path.basename(filename) for filename, _ in annotations]
    print("\n正在建立源图片索引...")
    source_index = build_source_index(SOURCE_DIRS, wanted_basenames)
    print(f"找到源图片: {len(source_index)} / {len(set(wanted_basenames))} 个唯一文件名")

    copied = 0
    already_exists = 0
    not_found = 0
    class_counts = Counter()

    for filename, label in annotations:
        basename = os.path.basename(filename)
        src_path = source_index.get(basename)
        if src_path is None:
            print(f"[未找到] {filename}")
            not_found += 1
            continue

        class_folder = LABEL_FOLDERS[label]
        dest_dir = os.path.join(OUTPUT_DIR, class_folder)
        dest_path, exists_same = get_unique_dest_path(dest_dir, basename, src_path)

        if exists_same:
            already_exists += 1
        else:
            shutil.copy2(src_path, dest_path)
            copied += 1
            if copied % 500 == 0:
                print(f"已复制 {copied} 张...")

        class_counts[class_folder] += 1

    print("\n复制完成")
    print(f"成功复制: {copied}")
    print(f"已存在跳过: {already_exists}")
    print(f"未找到: {not_found}")
    print("各类别数量:")
    for label, folder in LABEL_FOLDERS.items():
        print(f"  {label} {folder}: {class_counts[folder]}")


if __name__ == "__main__":
    copy_images()
