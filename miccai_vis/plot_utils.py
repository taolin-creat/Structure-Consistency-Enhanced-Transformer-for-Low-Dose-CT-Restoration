import os
import numpy as np
import matplotlib.pyplot as plt

from roi_utils import crop_roi


def plot_main_figure(images, case_name, save_dir):

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    methods = ["LDCT", "LITFormer", "Ours", "FDCT"]

    for i, m in enumerate(methods):
        axes[i].imshow(images[m], cmap='gray')
        axes[i].set_title(m)
        axes[i].axis("off")

    # ===== Error map =====
    error = np.abs(images["Ours"] - images["FDCT"])
    axes[4].imshow(error, cmap='hot')
    axes[4].set_title("Error")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, case_name.replace(".npy", "_main.png")))
    plt.close()


# -------------------------------------------------

def plot_roi_figure(images, case_name, save_dir):

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    methods = ["LDCT", "LITFormer", "Ours", "FDCT"]

    for i, m in enumerate(methods):

        roi = crop_roi(images[m])
        axes[i].imshow(roi, cmap='gray')
        axes[i].set_title(m)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, case_name.replace(".npy", "_roi.png")))
    plt.close()


# -------------------------------------------------

def plot_slice_consistency(root, folders, case_file, save_dir):

    slice_id = int(case_file.split("_")[-1].replace(".npy", ""))

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    methods = ["LITFormer", "Ours", "FDCT"]

    offsets = [-1, 0, 1]

    for row, offset in enumerate(offsets):

        new_slice = f"{slice_id+offset:04d}.npy"

        for col, method in enumerate(methods):

            path = os.path.join(root, folders[method], new_slice)

            if not os.path.exists(path):
                continue

            img = np.load(path)
            img = img.squeeze()

            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, case_file.replace(".npy", "_consistency.png")))
    plt.close()
