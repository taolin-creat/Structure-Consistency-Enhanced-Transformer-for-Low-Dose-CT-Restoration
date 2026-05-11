import os
import numpy as np
from pathlib import Path

from io_utils import load_case_slice
from window_utils import apply_window
from plot_utils import plot_main_figure, plot_roi_figure, plot_slice_consistency


# ======= 数据路径配置 =======
DATA_ROOT = r"D:\project\LIT-Former - change - 1\test_results\visual_compare"

METHOD_FOLDERS = {
    "LDCT": "LDCT",
    "FDCT": "FDCT",
    "LITFormer": "LITFormer",
    "Ours": "Ours"
}

SAVE_DIR = r"D:\project\LIT-Former - change - 1\test_results\experiment_name\paper_figs"

# ======= CT窗宽窗位 =======
WINDOW_MIN = -160
WINDOW_MAX = 240


def main():

    os.makedirs(SAVE_DIR, exist_ok=True)

    cases = sorted(os.listdir(os.path.join(DATA_ROOT, "Ours")))

    for case_file in cases:

        print("Processing:", case_file)

        images = {}

        for method, folder in METHOD_FOLDERS.items():

            path = os.path.join(DATA_ROOT, folder, case_file)
            img = load_case_slice(path)
            img = apply_window(img, WINDOW_MIN, WINDOW_MAX)

            images[method] = img

        # ===== 主对比图 =====
        plot_main_figure(images, case_file, SAVE_DIR)

        # ===== ROI 图 =====
        plot_roi_figure(images, case_file, SAVE_DIR)

        # ===== 连续切片一致性 =====
        plot_slice_consistency(DATA_ROOT, METHOD_FOLDERS, case_file, SAVE_DIR)


if __name__ == "__main__":
    main()
