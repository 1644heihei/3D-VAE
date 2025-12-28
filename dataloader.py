"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import torch
import numpy as np
import nibabel as ni
import os, shutil
import time
import random
import pandas as pd


def split_train_test(dir, ratio_test=0.15):
    if not os.path.exists(os.path.join(dir, "train")):
        os.mkdir(os.path.join(dir, "train"))
    if not os.path.exists(os.path.join(dir, "test")):
        os.mkdir(os.path.join(dir, "test"))

    images_list = [i for i in os.listdir(dir) if i.endswith(".nii")]

    random.shuffle(images_list)
    threshold = int(len(images_list) * ratio_test)
    train_list = images_list[:-threshold]
    test_list = images_list[-threshold:]

    for i in train_list:
        shutil.move(os.path.join(dir, i), os.path.join(dir, "train", i))
    for i in test_list:
        shutil.move(os.path.join(dir, i), os.path.join(dir, "test", i))


def save_data_to_csv(dir, z):
    pd.DataFrame(z).to_csv(dir, header=None, index=False)


"""
def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".raw")]
    if not filenames:
        print("No .raw files found in the specified directory.")
        return
    random.shuffle(filenames)
    n = 0
    while n < len(filenames):
        batch_image = []
        batch_stats = []  # 各画像の平均と標準偏差を記録
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                break
            filepath = os.path.join(path, filenames[i])
            image = np.fromfile(filepath, dtype=np.int16).reshape((80, 96, 80))  # reshapeは必要に応じて調整
            mean, std = image.mean(), image.std()  # 平均と標準偏差を計算
            if std == 0:  # 標準偏差が0の場合に対応
                std = 1
            image = (image - mean) / std  # 分散と平均を使用して正規化
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            batch_image.append(image)
            batch_stats.append((mean, std))  # 平均と標準偏差を記録
        n += batch_size
        if batch_image:
            batch_image = torch.cat(batch_image, axis=0)
            yield batch_image, batch_stats  # バッチデータと統計情報を返す

"""


def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".raw")]
    if not filenames:
        print("No .raw files found in the specified directory.")
        return
    random.shuffle(filenames)
    n = 0
    while n < len(filenames):
        batch_image = []
        batch_min_max = []  # 各画像の min/max を記録
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                break
            filepath = os.path.join(path, filenames[i])
            image = np.fromfile(filepath, dtype=np.int16).reshape(
                (80, 96, 80)
            )  # reshapeは必要に応じて調整
            original_min, original_max = (
                image.min(),
                image.max(),
            )  # 最小値・最大値を取得
            image = (image - original_min) / (original_max - original_min)  # 正規化
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            batch_image.append(image)
            batch_min_max.append((original_min, original_max))  # min/max を記録
        n += batch_size
        if batch_image:
            batch_image = torch.cat(batch_image, axis=0)
            yield batch_image, batch_min_max  # バッチデータと min/max を返す


#################### TEST #################
# start = time.time()
# for i in load_mri_images("./data", 2):
#     print(time.time()-start)
#     start = time.time()
#     print(i.shape)

# split_train_test("/home/ubuntu/Desktop/DuyPhuong/VAE/data")
