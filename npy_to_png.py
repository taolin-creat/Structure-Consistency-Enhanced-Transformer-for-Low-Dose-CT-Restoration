import os
import numpy as np
from pathlib import Path
from PIL import Image


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    把任意数值范围的数组线性归一化到 [0, 255] 并转换为 uint8，用于保存为灰度/彩色图像。
    """
    arr = arr.astype(np.float32)
    min_val = arr.min()
    max_val = arr.max()

    # 避免除零
    if max_val - min_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - min_val) / (max_val - min_val)  # [0, 1]
    arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def convert_npy_folder_to_png(src_folder: str, dst_folder: str):
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)

    if not src_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src_path}")

    # 创建输出文件夹
    dst_path.mkdir(parents=True, exist_ok=True)

    npy_files = list(src_path.glob("*.npy"))
    if not npy_files:
        print("源文件夹中没有找到 .npy 文件。")
        return

    for npy_file in npy_files:
        try:
            arr = np.load(npy_file)
        except Exception as e:
            print(f"加载失败: {npy_file.name}, 错误: {e}")
            continue

        # 根据维度处理
        if arr.ndim == 2:
            # 单通道灰度图
            img_arr = normalize_to_uint8(arr)
            mode = "L"
        elif arr.ndim == 3:
            # 可能是 HxWxC 或 CxHxW
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                # 假设是 CxHxW -> HxWxC
                arr = np.transpose(arr, (1, 2, 0))

            if arr.shape[-1] == 1:
                # 单通道 [H, W, 1] -> [H, W]
                arr = arr[..., 0]
                img_arr = normalize_to_uint8(arr)
                mode = "L"
            elif arr.shape[-1] in (3, 4):
                # RGB 或 RGBA
                img_arr = normalize_to_uint8(arr)
                mode = "RGB" if arr.shape[-1] == 3 else "RGBA"
            else:
                print(f"无法识别的通道数 {arr.shape[-1]}，跳过: {npy_file.name}")
                continue
        else:
            print(f"维度 {arr.ndim} 不支持，跳过: {npy_file.name}")
            continue

        # 生成输出文件名
        out_name = npy_file.stem + ".png"
        out_path = dst_path / out_name

        # 保存图片
        try:
            img = Image.fromarray(img_arr, mode=mode if img_arr.ndim > 2 else None)
            img.save(out_path)
            print(f"已保存: {out_path}")
        except Exception as e:
            print(f"保存失败: {out_path}, 错误: {e}")


if __name__ == "__main__":
    # ==== 在这里改你的路径 ====
    # 源文件夹：存放 .npy 的文件夹
    src_folder = r"D:\project\LIT-Former - change - 1\test_results\experiment_name\test-npy"       # 比如：r"D:\mydata\npy"
    # 目标文件夹：用于保存 .png 的新文件夹
    dst_folder = r"D:\project\LIT-Former - change - 1\test_results\experiment_name\test-png"       # 比如：r"D:\mydata\png"
    # ========================

    convert_npy_folder_to_png(src_folder, dst_folder)
