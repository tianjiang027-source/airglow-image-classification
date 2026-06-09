"""
兴隆数据集人工标注工具（仅标注，不复制图片）

功能：
1. 从数据集"F:\兴隆\数据集"每5张图片提取一张展示
2. 人工识别5个类别，通过GUI按钮进行标注
3. 仅记录标注结果，生成文本标注文件
4. 图片复制功能由独立脚本 copy_annotated_images.py 完成

注意：
- 人工未选择标注的图片不加入标注文件
- 无聚类，纯人工判断
- 不复制图片，只生成文本标注
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
from datetime import datetime

# ============ 配置 ============
SOURCE_DIR = r"F:\兴隆\数据集"
OUTPUT_BASE_DIR = r"F:\兴隆\训练集\train_xinglong"

# 5个类别名称（与 train_xinglong 子文件夹名称一致）
LABEL_NAMES = {
    0: "a_星空",
    1: "b_亮斑",
    2: "c_光条",
    3: "d_曝光",
    4: "e_黑暗",
}
CLUSTER_NUM = 5  # 类别数量

# 标注进度保存文件
PROGRESS_FILE = os.path.join(OUTPUT_BASE_DIR, "annotation_progress.json")
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def image_key(path):
    """使用相对路径作为图片唯一标识，支持递归遍历子文件夹。"""
    return os.path.normpath(os.path.relpath(path, SOURCE_DIR))


def load_progress():
    """加载标注进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "annotated": {},      # {filename: label_index}
        "skipped": [],       # [filename, ...] 人为跳过（人工判定不合格）的图片
        "current_index": 0,
        "last_update": ""
    }


def save_progress(progress):
    """保存标注进度"""
    progress["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def scan_all_images(source_dir, sample_step=5):
    """递归扫描文件夹下所有图片，按文件夹和文件名顺序每 sample_step 张取一张。"""
    all_images = []
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        files.sort()
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                all_images.append(os.path.join(root, f))
    all_images.sort(key=lambda p: os.path.normcase(os.path.relpath(p, source_dir)))
    sampled = all_images[::sample_step]
    return sampled, len(all_images)


def normalize_progress_keys(progress, image_paths):
    """兼容旧进度文件：把纯文件名转换成相对路径。"""
    basename_to_keys = {}
    for path in image_paths:
        basename_to_keys.setdefault(os.path.basename(path), []).append(image_key(path))

    def normalize_key(old_key):
        if os.path.dirname(old_key):
            return old_key
        matches = basename_to_keys.get(old_key, [])
        return matches[0] if len(matches) == 1 else old_key

    progress["annotated"] = {
        normalize_key(key): value
        for key, value in progress.get("annotated", {}).items()
    }
    progress["skipped"] = [
        normalize_key(key)
        for key in progress.get("skipped", [])
    ]
    return progress


class AnnotationGUI:
    """人工标注图形界面"""

    def __init__(self, image_paths, progress):
        self.image_paths = image_paths
        self.progress = progress
        self.annotated = progress.get("annotated", {})
        self.skipped = set(progress.get("skipped", []))
        self.current_idx = progress.get("current_index", 0)

        # 确保索引有效
        if self.current_idx >= len(self.image_paths):
            self.current_idx = 0

        self.photo_image = None
        self.canvas_image_id = None

        self.window = tk.Tk()
        self.window.title("兴隆数据集人工标注工具（仅标注）")
        self.window.geometry("1200x850")
        self.window.minsize(900, 700)

        self._create_widgets()
        self.window.update_idletasks()
        self._show_image()

    def _create_widgets(self):
        # 顶部信息栏
        info_frame = ttk.Frame(self.window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 11))
        self.info_label.pack(side=tk.LEFT)

        # 进度百分比显示
        self.progress_label = ttk.Label(info_frame, text="", font=('Arial', 11))
        self.progress_label.pack(side=tk.RIGHT)

        # 图像显示区域
        img_frame = ttk.Frame(self.window)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(img_frame, bg='#2a2a2a')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 类别按钮区域
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.label_buttons = {}
        for i in range(CLUSTER_NUM):
            name = LABEL_NAMES[i]
            btn = ttk.Button(
                btn_frame,
                text=f"{i + 1}. {name}",
                command=lambda x=i: self.annotate_and_next(x)
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5, ipadx=8)
            self.label_buttons[i] = btn

        # 跳过按钮
        skip_btn = ttk.Button(
            btn_frame,
            text="跳过 (S)",
            command=self.skip_image
        )
        skip_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 底部控制栏
        ctrl_frame = ttk.Frame(self.window)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(ctrl_frame, text="上一张 (←)", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="下一张 (→)", command=self.next_image).pack(side=tk.LEFT, padx=5)

        # 完成并导出按钮
        export_btn = ttk.Button(
            ctrl_frame,
            text="完成并导出标注文件",
            command=self.finish_and_export
        )
        export_btn.pack(side=tk.RIGHT, padx=5)

        save_btn = ttk.Button(
            ctrl_frame,
            text="保存进度",
            command=self.save_progress_gui
        )
        save_btn.pack(side=tk.RIGHT, padx=5)

        # 统计信息
        self.stats_label = ttk.Label(self.window, text="", font=('Arial', 10))
        self.stats_label.pack(pady=3)

        # 键盘快捷键
        self.window.bind('<Left>', lambda e: self.prev_image())
        self.window.bind('<Right>', lambda e: self.next_image())
        self.window.bind('<space>', lambda e: self.next_image())
        self.window.bind('s', lambda e: self.skip_image())
        self.window.bind('S', lambda e: self.skip_image())
        for i in range(CLUSTER_NUM):
            self.window.bind(str(i + 1), lambda e, x=i: self.annotate_and_next(x))

    def _show_image(self):
        if self.current_idx >= len(self.image_paths):
            self.info_label.config(text="所有图片已处理完毕！")
            self.stats_label.config(text="")
            return

        path = self.image_paths[self.current_idx]
        img_key = image_key(path)
        filename = os.path.basename(path)
        is_skipped = img_key in self.skipped
        is_annotated = img_key in self.annotated

        # 显示信息
        status_parts = []
        if is_annotated:
            status_parts.append(f"已标注: {LABEL_NAMES.get(self.annotated[img_key], '未知类别')}")
        if is_skipped:
            status_parts.append("已跳过")
        status_str = " | ".join(status_parts) if status_parts else "待标注"

        total = len(self.image_paths)
        self.info_label.config(
            text=f"第 {self.current_idx + 1} / {total} 张 | {status_str} | {img_key}"
        )

        # 统计
        from collections import Counter
        annotated_count = len(self.annotated)
        skipped_count = len(self.skipped)
        pending_count = sum(
            1 for p in self.image_paths
            if image_key(p) not in self.annotated
            and image_key(p) not in self.skipped
        )
        # 各类别数量
        counter = Counter(self.annotated.values())
        per_class = " | ".join(
            f"{LABEL_NAMES[i]}: {counter.get(i, 0)}" for i in range(CLUSTER_NUM)
        )
        self.stats_label.config(
            text=f"已标注: {annotated_count} | 已跳过: {skipped_count} | 待处理: {pending_count} | {per_class}"
        )

        # 加载并显示图像
        try:
            if self.canvas_image_id:
                self.canvas.delete(self.canvas_image_id)

            img = Image.open(path)

            # 处理不同格式
            if img.mode in ('I', 'I;16', 'I;16L', 'I;16B'):
                import numpy as np
                img_array = np.array(img)
                if img_array.dtype == np.uint16:
                    img_array = ((img_array - img_array.min()) /
                                (img_array.max() - img_array.min() + 1e-10) * 255).astype(np.uint8)
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                img = Image.fromarray(img_array)
            elif img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # 裁剪中间 512x512 区域
            img_w, img_h = img.size
            crop_size = 512
            left = (img_w - crop_size) // 2
            top = (img_h - crop_size) // 2
            img = img.crop((left, top, left + crop_size, top + crop_size))

            # 缩放以适应窗口
            canvas_w = max(self.canvas.winfo_width(), 900)
            canvas_h = max(self.canvas.winfo_height(), 600)
            scale = min(canvas_w / crop_size, canvas_h / crop_size)
            new_w = int(crop_size * scale)
            new_h = int(crop_size * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(img)

            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.canvas_image_id = self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)

        except Exception as e:
            self.info_label.config(text=f"加载图像失败: {e} | {filename}")

    def annotate_and_next(self, label_index):
        """标注当前图片并自动下一张"""
        if self.current_idx >= len(self.image_paths):
            return

        img_key = image_key(self.image_paths[self.current_idx])

        # 记录标注结果
        self.annotated[img_key] = label_index
        # 从跳过列表移除（如果之前跳过）
        self.skipped.discard(img_key)
        print(f"[标注] {img_key} -> {LABEL_NAMES[label_index]}")

        # 自动显示顺序列表中的下一张
        self._move_to_next()

    def skip_image(self):
        """跳过当前图片（人工判定不合格，不标注）"""
        if self.current_idx >= len(self.image_paths):
            return
        img_key = image_key(self.image_paths[self.current_idx])
        self.skipped.add(img_key)
        print(f"[跳过] {img_key}")
        self._move_to_next()

    def _move_to_next(self):
        """移动到顺序列表中的下一张图片，不跳过已标注图片。"""
        self.current_idx += 1
        self._show_image()

    def next_image(self):
        """手动下一张（不标注）"""
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self._show_image()

    def prev_image(self):
        """上一张"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_image()

    def save_progress_gui(self):
        """保存进度"""
        self.progress["annotated"] = self.annotated
        self.progress["skipped"] = list(self.skipped)
        self.progress["current_index"] = self.current_idx
        save_progress(self.progress)
        messagebox.showinfo("保存", f"进度已保存\n标注: {len(self.annotated)} 张\n跳过: {len(self.skipped)} 张")

    def finish_and_export(self):
        """完成标注并导出标注文件"""
        # 保存进度
        self.progress["annotated"] = self.annotated
        self.progress["skipped"] = list(self.skipped)
        self.progress["current_index"] = self.current_idx
        save_progress(self.progress)

        # 数字标签到字母的映射（与子文件夹名称 a_*, b_*, c_*, d_*, e_* 对应）
        label_to_char = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        # 生成标注文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_BASE_DIR, f"annotations_{timestamp}.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 兴隆数据集标注文件\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 总标注数: {len(self.annotated)}\n")
            f.write(f"# 总跳过数: {len(self.skipped)}\n")
            f.write(f"# 格式: filename label_char class_name\n")
            f.write(f"# 标签对应: ")
            for idx, name in LABEL_NAMES.items():
                f.write(f"{label_to_char[idx]}={name}  ")
            f.write(f"\n\n")

            for filename, label_idx in sorted(self.annotated.items()):
                char = label_to_char.get(label_idx, str(label_idx))
                f.write(f"{filename} {char} {LABEL_NAMES.get(label_idx, '未知类别')}\n")

        # 生成简化标注文件（仅 filename + 标签字母）
        simple_file = os.path.join(OUTPUT_BASE_DIR, "annotations_simple.txt")
        with open(simple_file, 'w', encoding='utf-8') as f:
            for filename, label_idx in sorted(self.annotated.items()):
                char = label_to_char.get(label_idx, str(label_idx))
                f.write(f"{filename} {char}\n")

        # 生成跳过文件
        if self.skipped:
            skipped_file = os.path.join(OUTPUT_BASE_DIR, "skipped_images.txt")
            with open(skipped_file, 'w', encoding='utf-8') as f:
                for filename in sorted(self.skipped):
                    f.write(f"{filename}\n")

        # 统计各类别数量
        from collections import Counter
        counter = Counter(self.annotated.values())
        stats_lines = "\n".join(
            [f"  类别{i} ({LABEL_NAMES[i]}): {counter.get(i, 0)} 张" for i in range(CLUSTER_NUM)]
        )

        msg = (
            f"标注完成！\n\n"
            f"已标注: {len(self.annotated)} 张\n"
            f"已跳过: {len(self.skipped)} 张\n\n"
            f"各类别统计:\n{stats_lines}\n\n"
            f"标注文件:\n{output_file}\n\n"
            f"简化标注文件:\n{simple_file}\n\n"
            f"下一步: 运行 copy_annotated_images.py 将图片复制到子文件夹。"
        )
        messagebox.showinfo("导出完成", msg)

    def run(self):
        self.window.mainloop()


def main():
    print("=" * 60)
    print("兴隆数据集人工标注工具（仅标注）")
    print("=" * 60)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # 加载进度
    progress = load_progress()
    print(f"已加载进度: 标注 {len(progress.get('annotated', {}))} 张, 跳过 {len(progress.get('skipped', []))} 张")

    # 扫描图片（每5张取1张）
    print("\n正在扫描数据集...")
    all_images, total_images = scan_all_images(SOURCE_DIR)
    print(f"数据集总计: {total_images} 张原始图片")
    print(f"每5张取1张后: {len(all_images)} 张待处理图片")

    if not all_images:
        print("未找到图片文件！")
        return

    progress = normalize_progress_keys(progress, all_images)

    # 过滤掉已标注和已跳过的
    remaining = []
    annotated_set = set(progress.get("annotated", {}).keys())
    skipped_set = set(progress.get("skipped", []))

    for path in all_images:
        img_key = image_key(path)
        if img_key not in annotated_set and img_key not in skipped_set:
            remaining.append(path)

    print(f"待处理: {len(remaining)} 张")

    # 保持扫描顺序，不把未处理图片提前；标注后会按列表顺序进入下一张。
    ordered_paths = all_images

    # 启动GUI
    print(f"\n启动标注界面...")
    print("使用方法:")
    print("  1-5: 标注为对应类别（自动下一张）")
    print("  S:   跳过当前图片（不标注）")
    print("  ←/→: 手动切换图片")
    print("  空格: 下一张")
    print("  保存: 保存进度")
    print("  导出: 完成并导出标注文件")
    print("注意: 本工具不复制图片，图片复制由 copy_annotated_images.py 完成")

    gui = AnnotationGUI(ordered_paths, progress)
    gui.run()

    print("\n标注工具已关闭。")


if __name__ == "__main__":
    main()
