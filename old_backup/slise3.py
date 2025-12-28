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

if not os.path.exists("save_image7"):
    os.mkdir("save_image7")
if not os.path.exists("dataset"):
    os.mkdir("dataset")
if not os.path.exists("asset"):
    os.mkdir("asset")

    
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

class DownSizePairImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=256, small_size=64, **kwds):
        super().__init__(root, transform=transform, **kwds)
        #self.large_resizer = transforms.Scale(large_size)
        #self.small_resizer = transforms.Scale(small_size)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)
        
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)
        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)
        return small_img, large_img
    
DimSize1 = [80,96,80]
img17h = "saved_raw_data/output_step_3000_image_0.raw"
#img17h = "output_raw_data2/1000.raw"
img17l = "saved_raw_data/input_step_500_image_0.raw"
img18h = "saved_raw_data/input_step_500_image_0.raw."
img18l = "saved_raw_data/input_step_500_image_0.raw"

def data_load(img17h,img17l,img18h,img18l,DimSize1):
    DimSize = DimSize1
    tmp_7h = img17h
    tmp_7l = img17l
    tmp_8h = img18h
    tmp_8l = img18l

    fname = tmp_7h
    HR_797_1 = np.fromfile(fname, '<h')
    #print(HR_797_1.shaprawe)
    HR_797 = HR_797_1.reshape([DimSize[2],DimSize[1],DimSize[0]])
    #data = data.transpose(2,1,0)
    #HR_797=HR_797_1
   
    print(HR_797.shape)
    print(HR_797[0,0,5])

    fname = tmp_7l
    LR_797_1 = np.fromfile(fname, '<h')
    LR_797 = LR_797_1.reshape([DimSize[2],DimSize[1],DimSize[0]])
    #data = data.transpose(2,1,0)
    
    print(LR_797.shape)

    fname = tmp_8h
    HR_835_1 = np.fromfile(fname, '<h')
    HR_835 = HR_835_1.reshape([DimSize[2],DimSize[1],DimSize[0]])
    #data = data.transpose(2,1,0)
   
    print(HR_835.shape)

    fname = tmp_8l
    LR_835_1 = np.fromfile(fname, '<h')
    LR_835 = LR_835_1.reshape([DimSize[2],DimSize[1],DimSize[0]])
    #data = data.transpose(2,1,0)
   
    print(LR_835.shape)
    

    return HR_797,LR_797,HR_835,LR_835

def data_cutting_LR753_3(imL797,y,DimSize):
    im = np.empty([0,96,80],int)
    for i in range(80):
                im=imL797[:,:,i]
                cv2.imwrite(f"save_image7/{y}.jpg",im)
                y+=1
    
    return y

imH797,imL797,imH835,imL835=data_load(img17h,img17l,img18h,img18l,DimSize1)
print(imH797.shape)
y=data_cutting_LR753_3(imH797,0,DimSize1)
