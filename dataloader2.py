import torch
import numpy as np
import os, shutil
import random
import pandas as pd 

def split_train_test(dir, ratio_test=0.15):
    if not os.path.exists(os.path.join(dir, "train")): os.mkdir(os.path.join(dir, "train"))
    if not os.path.exists(os.path.join(dir, "test")): os.mkdir(os.path.join(dir, "test"))
    
    images_list = [i for i in os.listdir(dir) if i.endswith(".raw")]

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

def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".raw")]
    random.shuffle(filenames, random.random)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                break
            # Load raw file
            filepath = os.path.join(path, filenames[i])
            image = np.fromfile(filepath, dtype=np.int16).reshape((80, 96, 80))  # Adjust shape if necessary
            # Normalize data
            image = (image - image.min()) / (image.max() - image.min())
            image = torch.tensor(image, dtype=torch.float32)
            image = torch.reshape(image, (1, 1, 80, 96, 80))
            batch_image.append(image)
        n += batch_size
        batch_image = torch.cat(batch_image, axis=0)
        yield batch_image
