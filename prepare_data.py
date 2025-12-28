import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch
import os
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from torchvision.utils import save_image

from torchvision.models.vgg import vgg16
cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available!')



    
import torch
from torch import nn, optim
from torch.utils.data import (Dataset, 
                              DataLoader,
                              TensorDataset)
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

DimSize1 = [1800,1752,300]
img17h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240422180726_sample)_scan(20240422182550_still)/scan(20240422182550_still)_recon(20240422190026.618_1533.0_0.100_-15.0_15.0).binary/volume_756.raw"
img17l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240422180726_sample)_scan(20240422181104_still)/scan(20240422181104_still)_recon(20240422183101.003_decimation5_1533.5_0.100_-15.0_15.0).binary/volume_756.raw"
img18h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240422180726_sample)_scan(20240422182550_still)/scan(20240422182550_still)_recon(20240422190026.618_1533.0_0.100_-15.0_15.0).binary/volume_797.raw"
img18l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240422180726_sample)_scan(20240422181104_still)/scan(20240422181104_still)_recon(20240422183101.003_decimation5_1533.5_0.100_-15.0_15.0).binary/volume_797.raw"

DimSize2 = [1740,1320,300]
img27h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423130527_Palm001)_scan(20240423131843_still)/scan(20240423131843_still)_recon(20240423151216.758_1531.9_0.100_-15.0_15.0).binary/volume_756.raw"
img27l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423130527_Palm001)_scan(20240423130803_still)/scan(20240423130803_still)_recon(20240423145738.639_decimation5_1532.4_0.100_-15.0_15.0).binary/volume_756.raw"
img28h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423130527_Palm001)_scan(20240423131843_still)/scan(20240423131843_still)_recon(20240423151216.758_1531.9_0.100_-15.0_15.0).binary/volume_797.raw"
img28l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423130527_Palm001)_scan(20240423130803_still)/scan(20240423130803_still)_recon(20240423145738.639_decimation5_1532.4_0.100_-15.0_15.0).binary/volume_797.raw"

DimSize3 = [1740,1420,300]
img37h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423130527_Palm001)_scan(20240423133408_still)/scan(20240423133408_still)_recon(20240423151224.592_1532.3_0.100_-15.0_15.0).binary/volume_756.raw"
img37l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423130527_Palm001)_scan(20240423132259_still)/scan(20240423132259_still)_recon(20240423145827.396_decimation5_1532.7_0.100_-15.0_15.0).binary/volume_756.raw"
img38h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423130527_Palm001)_scan(20240423133408_still)/scan(20240423133408_still)_recon(20240423151224.592_1532.3_0.100_-15.0_15.0).binary/volume_797.raw"
img38l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423130527_Palm001)_scan(20240423132259_still)/scan(20240423132259_still)_recon(20240423145827.396_decimation5_1532.7_0.100_-15.0_15.0).binary/volume_797.raw"

DimSize4 = [1800,1300,300]
img47h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423134040_Palm002)_scan(20240423135542_still)/scan(20240423135542_still)_recon(20240423151233.639_1533.5_0.100_-15.0_15.0).binary/volume_756.raw"
img47l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423134040_Palm002)_scan(20240423134452_still)/scan(20240423134452_still)_recon(20240423145903.498_decimation5_1533.3_0.100_-15.0_15.0).binary/volume_756.raw"
img48h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423134040_Palm002)_scan(20240423135542_still)/scan(20240423135542_still)_recon(20240423151233.639_1533.5_0.100_-15.0_15.0).binary/volume_797.raw"
img48l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423134040_Palm002)_scan(20240423134452_still)/scan(20240423134452_still)_recon(20240423145903.498_decimation5_1533.3_0.100_-15.0_15.0).binary/volume_797.raw"

DimSize5 = [1800,1220,300]
img57h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423134040_Palm002)_scan(20240423141005_still)/scan(20240423141005_still)_recon(20240423151238.760_1533.5_0.100_-15.0_15.0).binary/volume_756.raw"
img57l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423134040_Palm002)_scan(20240423135956_still)/scan(20240423135956_still)_recon(20240423151024.728_decimation5_1532.8_0.100_-15.0_15.0).binary/volume_756.raw"
img58h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423134040_Palm002)_scan(20240423141005_still)/scan(20240423141005_still)_recon(20240423151238.760_1533.5_0.100_-15.0_15.0).binary/volume_797.raw"
img58l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423134040_Palm002)_scan(20240423135956_still)/scan(20240423135956_still)_recon(20240423151024.728_decimation5_1532.8_0.100_-15.0_15.0).binary/volume_797.raw"

DimSize6 = [1800,1832,300]
img67h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423141327_Palm003)_scan(20240423142956_still)/scan(20240423142956_still)_recon(20240423151242.879_1533.9_0.100_-15.0_15.0).binary/volume_756.raw"
img67l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423141327_Palm003)_scan(20240423141503_still)/scan(20240423141503_still)_recon(20240423151055.461_decimation5_1533.1_0.100_-15.0_15.0).binary/volume_756.raw"
img68h = "C:/Users/FUJII/Desktop/20240423/IQL1/study(20240423141327_Palm003)_scan(20240423142956_still)/scan(20240423142956_still)_recon(20240423151242.879_1533.9_0.100_-15.0_15.0).binary/volume_797.raw"
img68l = "C:/Users/FUJII/Desktop/20240423/IQL3/study(20240423141327_Palm003)_scan(20240423141503_still)/scan(20240423141503_still)_recon(20240423151055.461_decimation5_1533.1_0.100_-15.0_15.0).binary/volume_797.raw"

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

output_dir = "output_raw_data2"
os.makedirs(output_dir, exist_ok=True)  # ディレクトリがない場合は作成

# 並列処理で実行する関数
def process_cut(i, j, k, slidex, slidey, slidez, data, cutx, cuty, cutz):
    d_slide = i * slidez
    h_slide = j * slidey
    w_slide = k * slidex
    # 32x32x32 のデータを切り出し
    return data[:, :, d_slide:80 + d_slide, h_slide:96 + h_slide, w_slide:80 + w_slide]

def cut(y, DimSize, img17h, img17l, img18h, img18l):
    # パラメータ設定
    numa = 80
    numa2=96
    dimx, dimy, dimz = DimSize
    
    # z軸の範囲を36~100に限定
    z_start, z_end = 20, 120
    cutx, cuty, cutz = 22, 18, (z_end - z_start) // (numa)  # z軸に合わせて計算

    numx = (dimx - ((dimx - numa) % (cutx - 1)))
    numy = (dimy - ((dimy - numa2) % (cuty )))
    numz = z_end - z_start

    slidex = (numx - numa) // (cutx - 1)
    slidey = (numy - numa2) // (cuty - 1)
    slidez = (numz - numa) // (cutz )

    # データ読み込みとリシェイプ
    def load_and_prepare(filename):
        data = np.fromfile(filename, '<h').reshape([dimz, dimy, dimx])
        # z軸の切り出し (36~100 の範囲)
        data = data[z_start:z_end, :, :]
        return data[:numz, :numy, :numx][np.newaxis, np.newaxis, :, :, :]

    HR_797 = load_and_prepare(img17h)
    LR_797 = load_and_prepare(img17l)
    HR_835 = load_and_prepare(img18h)
    LR_835 = load_and_prepare(img18l)

    # 空配列の事前確保
    total_cuts = cutx * cuty * cutz
    tmp_LR_cutdata797 = np.empty((total_cuts, 1, 80, 96, 80), dtype=int)
    tmp_HR_cutdata797 = np.empty((total_cuts, 1, 80, 96, 80), dtype=int)
    tmp_LR_cutdata835 = np.empty((total_cuts, 1, 80, 96, 80), dtype=int)
    tmp_HR_cutdata835 = np.empty((total_cuts, 1, 80, 96, 80), dtype=int)

    # 並列処理でデータを切り出し
    def parallel_cut(data, tmp_cutdata):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_cut, i, j, k, slidex, slidey, slidez, data, cutx, cuty, cutz)
                for i in range(cutz)
                for j in range(cuty)
                for k in range(cutx)
            ]
            for idx, future in enumerate(futures):
                tmp_cutdata[idx] = future.result()

    parallel_cut(LR_797, tmp_LR_cutdata797)
    parallel_cut(HR_797, tmp_HR_cutdata797)
    parallel_cut(LR_835, tmp_LR_cutdata835)
    parallel_cut(HR_835, tmp_HR_cutdata835)

    # データ保存
    def save_cutdata(tmp_cutdata, prefix):
        nonlocal y
        for i in range(tmp_cutdata.shape[0]):
            data = tmp_cutdata[i, 0, :, :, :]
            if np.all(data == 0):  # データがすべてゼロの場合はスキップ
                print(f"Skipping empty data at index {i}")
                continue

            data = np.clip(data, 0, 255)
            filename = os.path.join(output_dir, f"{y}.raw")
            data.astype(np.int16).tofile(filename)
            y += 1

    save_cutdata(tmp_LR_cutdata797, "LR_797")
    save_cutdata(tmp_HR_cutdata797, "HR_797")
    save_cutdata(tmp_LR_cutdata835, "LR_835")
    save_cutdata(tmp_HR_cutdata835, "HR_835")

    print(f"保存が完了しました: {output_dir}")
    return y


# データの処理
y = 0
y = cut(y, [1800, 1752, 300], img17h, img17l, img18h, img18l)
y = cut(y, [1740, 1320, 300], img27h, img27l, img28h, img28l)
y = cut(y, [1740, 1420, 300], img37h, img37l, img38h, img38l)
y = cut(y, [1800, 1300, 300], img47h, img47l, img48h, img48l)
y = cut(y, [1800, 1220, 300], img57h, img57l, img58h, img58l)
y = cut(y, [1800, 1832, 300], img67h, img67l, img68h, img68l)
