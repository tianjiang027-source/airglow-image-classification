import os
import sys
from collections import Counter


sys.stdout.reconfigure(encoding="utf-8")

TRAIN_DIR = r"F:\兴隆\训练集\class_train\3"
OUTPUT_FILE = r"F:\兴隆\训练集\class_train\3\classification_results.txt"

CLASS_FOLDERS = {
    "a": "a_星空",
    "b": "b_云层",
    "c": "c_光球光条",
    "d": "d_曝光",
    "e": "e_黑暗",
    "f": "f_异常",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    results = []
    counts = Counter()

    for label, folder in CLASS_FOLDERS.items():
        class_path = os.path.join(TRAIN_DIR, folder)
        if not os.path.isdir(class_path):
            print(f"[警告] 类别文件夹不存在: {class_path}")
            continue

        for filename in sorted(os.listdir(class_path)):
            file_path = os.path.join(class_path, filename)
            if not os.path.isfile(file_path):
                continue
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            results.append(f"{filename} {label}")
            counts[label] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("\n".join(results))
        if results:
            out.write("\n")

    print(f"Generated: {OUTPUT_FILE}")
    print(f"Total entries: {len(results)}")
    print("Class distribution:")
    for label, folder in CLASS_FOLDERS.items():
        print(f"  {label} {folder}: {counts[label]} files")


if __name__ == "__main__":
    main()
