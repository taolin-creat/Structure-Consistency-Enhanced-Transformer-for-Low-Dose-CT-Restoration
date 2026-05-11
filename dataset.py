#dataset
import os, glob, shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from util import transforms
import ipdb
import random
random.seed(42)

import re  # ✅ 加上这行

def natural_key(s):
    """用于自然排序的key函数，例如IM1、IM2、IM10按数字排"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    # tmplist.sort() # sorting the found list
    tmplist.sort(key=natural_key)  # ✅ 改用自然排序
    return tmplist

def random_sample(input_list, sample_size):
    if sample_size > len(input_list):
        sample_size = len(input_list)
    return random.sample(input_list, sample_size)


# class Mayo_Dataset(Dataset):
#     def __init__(self, opt,transforms=None):
#         #ipdb.set_trace()
#         self.transforms = transforms
#         #hu_min, hu_max = hu_range
#         self.phase=opt.phase
#         self.mirror_padding=opt.mirror_padding
#
#         self.q_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/quarter/*')
#         self.f_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/full/*')
#
#
#     def __getitem__(self, index):
#         f_data=np.load(self.f_path_list[index]).astype(np.float32)
#         q_data = np.load(self.q_path_list[index]).astype(np.float32)
#
#         if self.transforms is not None:
#             f_data = self.transforms[pytorch_ssim_package](f_data)
#             q_data = self.transforms[0](q_data)
#         return q_data, f_data
#
#     def __len__(self):
#         return len(self.q_path_list)
#
#

class Mayo_Dataset(Dataset):
    def __init__(self, opt, transforms=None):
        self.transforms = transforms
        self.phase = opt.phase
        self.mirror_padding = opt.mirror_padding

        self.q_path_list = sorted_list(opt.dataroot + '/' + opt.phase + '/quarter/*')
        self.f_path_list = sorted_list(opt.dataroot + '/' + opt.phase + '/full/*')

        # ====== 调试打印 ======
        print(f"[Mayo_Dataset] phase={self.phase}")
        print(f"[Mayo_Dataset] quarter 路径模式: {opt.dataroot}/{opt.phase}/quarter/*")
        print(f"[Mayo_Dataset] full 路径模式: {opt.dataroot}/{opt.phase}/full/*")
        print(f"[Mayo_Dataset] 找到 quarter 文件数: {len(self.q_path_list)}")
        print(f"[Mayo_Dataset] 找到 full 文件数: {len(self.f_path_list)}")

        if len(self.q_path_list) == 0 or len(self.f_path_list) == 0:
            print("⚠️ 警告: 数据集为空，请检查路径和文件！")

    def __getitem__(self, index):
        f_data = np.load(self.f_path_list[index]).astype(np.float32)
        q_data = np.load(self.q_path_list[index]).astype(np.float32)

        if self.transforms is not None:
            f_data = self.transforms[1](f_data)
            q_data = self.transforms[0](q_data)
        return q_data, f_data

    def __len__(self):
        return len(self.q_path_list)
