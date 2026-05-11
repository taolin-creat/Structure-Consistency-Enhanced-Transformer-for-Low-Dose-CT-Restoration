import numpy as np


def load_case_slice(path):

    arr = np.load(path)

    # ===== 自动降维 =====
    if arr.ndim == 4:        # [B,C,H,W]
        arr = arr[0, 0]

    elif arr.ndim == 3:
        arr = arr[0]

    return arr

