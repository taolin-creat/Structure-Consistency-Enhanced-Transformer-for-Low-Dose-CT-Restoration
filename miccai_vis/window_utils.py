import numpy as np


def apply_window(img, w_min, w_max):

    img = img * 3000 - 1000   # 反归一化 HU
    img = np.clip(img, w_min, w_max)

    img = (img - w_min) / (w_max - w_min)

    return img
