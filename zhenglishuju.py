import os
import shutil
from pathlib import Path

# ===== 修改这里 =====
litformer_dir = r"D:\project\LIT-Former - tl - 1\test_results\experiment_name\test-npy"
ours_dir = r"D:\project\LIT-Former - change - 1\test_results\experiment_name\test-npy"

output_root = r"D:\project\LIT-Former - change - 1\test_results\visual_compare"
# =====================


folders = {
    "LDCT": Path(output_root) / "LDCT",
    "FDCT": Path(output_root) / "FDCT",
    "LITFormer": Path(output_root) / "LITFormer",
    "Ours": Path(output_root) / "Ours"
}

for f in folders.values():
    f.mkdir(parents=True, exist_ok=True)


# ===== 从 Ours 复制 input / gt =====
for file in os.listdir(ours_dir):

    src = Path(ours_dir) / file

    if file.endswith("_input.npy"):
        new_name = file.replace("_input", "")
        shutil.copy(src, folders["LDCT"] / new_name)

    elif file.endswith("_gt.npy"):
        new_name = file.replace("_gt", "")
        shutil.copy(src, folders["FDCT"] / new_name)

    elif file.endswith("_pred.npy"):
        new_name = file.replace("_pred", "")
        shutil.copy(src, folders["Ours"] / new_name)


# ===== 从 LITFormer 复制 pred =====
for file in os.listdir(litformer_dir):

    if file.endswith("_pred.npy"):
        src = Path(litformer_dir) / file
        new_name = file.replace("_pred", "")
        shutil.copy(src, folders["LITFormer"] / new_name)


print("✅ 数据整理完成")
