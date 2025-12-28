"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import nibabel as ni
import numpy as np
import os, glob
import torch 
import csv
from tqdm import tqdm

import model3
import loss
import dataloader

##---------Settings--------------------------
batch_size = 8
lrate = 0.001
epochs = 30
weight_decay = 5e-7
##############
path_data = "output_raw_data2"
path2save = "./checkpoint/vae_t1/model_vae_epoch_{}.pt"
dir_info = './infor'       
f = open(os.path.join(dir_info,'model_vae_t1.csv'),'w',newline='')


####################
verbose = True
log = print if verbose else lambda *x, **i: None
np.random.seed(10)
torch.manual_seed(10)
###################
criterion_rec = loss.L1Loss()
criterion_dis = loss.KLDivergence()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(" GPU is activated" if device else " CPU is activated")
no_images = len(glob.glob(path_data + "/*.raw"))
print("Number of MRI images: ", no_images)
if __name__ == "__main__":
    vae_model = model3.VAE()
    vae_model.to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lrate, weight_decay=weight_decay)
    step = 0  # グローバルステップカウンタ
    save_dir = "./saved_raw_data"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        loss_rec_epoch, loss_KL_epoch, total_loss_epoch = 0, 0, 0

        # training phase
        vae_model.train()
        for batch_images, batch_min_max in tqdm(dataloader.load_mri_images(path_data, batch_size)):
            optimizer.zero_grad()
            batch_images = batch_images.to(device)
            #print(f"Batch input min: {batch_images.min()}, max: {batch_images.max()}")

            y, z_mean, z_log_sigma = vae_model(batch_images)
            """
            print(f"Output min: {y.min()}, max: {y.max()}")
            print(f"z_mean: min={z_mean.min()}, max={z_mean.max()}")
            print(f"z_log_sigma: min={z_log_sigma.min()}, max={z_log_sigma.max()}")
            """

            kl_weight = 0.01

            

            # 損失計算
            loss_rec_batch = criterion_rec(batch_images, y)
            loss_KL_batch = criterion_dis(z_mean, z_log_sigma)
            total_loss_batch = loss_rec_batch + kl_weight * loss_KL_batch

            # 最適化
            total_loss_batch.backward()
            optimizer.step()

            

            loss_rec_epoch += loss_rec_batch.item() * batch_images.shape[0]
            loss_KL_epoch += loss_KL_batch.item() * batch_images.shape[0]
            total_loss_epoch += total_loss_batch.item() * batch_images.shape[0]
         
            if step%30==0:
                log_file = "train2.txt"

                with open(log_file, "a") as f:
                    f.write(f"Epoch {epoch+1}, Step {step}: Reconstruction Loss={loss_rec_batch.item()}, "
                            f"KL Loss={loss_KL_batch.item()}, Total Loss={total_loss_batch.item()}\n")
                    f.write(f"Output stats: min={y.min().item()}, max={y.max().item()}, mean={y.mean().item()}\n")

            """

            # 正規化解除して保存: 入力データと出力データを RAW形式で保存
            if step % 500 == 0:
                batch_images_np = batch_images.cpu().numpy()
                output_images_np = y.cpu().detach().numpy()

                for i, (img, (mean, std)) in enumerate(zip(batch_images_np, batch_stats)):
                    # 入力データの正規化解除
                    input_denormalized = (img[0] * std) + mean
                    # 出力データの正規化解除
                    output_denormalized = (output_images_np[i, 0] * std) + mean

                    # RAW形式で保存（3D形状を保持）
                    input_save_path = os.path.join(save_dir, f"input_step_{step}_image_{i}.raw")
                    output_save_path = os.path.join(save_dir, f"output_step_{step}_image_{i}.raw")
                    input_denormalized.astype(np.int16).tofile(input_save_path)
                    output_denormalized.astype(np.int16).tofile(output_save_path)


            """
            # 正規化解除して保存: 入力データと出力データを RAW形式で保存
            if step % 500 == 0:



                batch_images_np = batch_images.cpu().numpy()
                output_images_np = y.cpu().detach().numpy()

                for i, (img, (original_min, original_max)) in enumerate(zip(batch_images_np, batch_min_max)):
                    # 入力データの正規化解除
                    input_denormalized = (img[0] * (original_max - original_min)) + original_min
                    # 出力データの正規化解除
                    output_denormalized = (output_images_np[i, 0] * (original_max - original_min)) + original_min

                    # RAW形式で保存（3D形状を保持）
                    input_save_path = os.path.join(save_dir, f"input_step_{step}_image_{i}.raw")
                    output_save_path = os.path.join(save_dir, f"output_step_{step}_image_{i}.raw")
                    input_denormalized.astype(np.int16).tofile(input_save_path)
                    output_denormalized.astype(np.int16).tofile(output_save_path)

                    #print(f"Saving RAW data with shape: {input_denormalized.shape}")


                #print(f"Saved RAW input and output data at step {step}")
            

            step += 1  # ステップをカウント

        # モデル保存
        log_info = (epoch + 1, epochs, loss_rec_epoch/no_images, loss_KL_epoch/no_images)
        log('%d/%d  Reconstruction Loss %.3f| KL Loss %.3f' % log_info)
        torch.save(vae_model, path2save.format(epoch + 1))

        # CSVに書き込み
        #writer = csv.writer(f)
        #writer.writerow([epoch + 1, '{:04f}'.format(loss_rec_epoch / no_images),
         #                               '{:04f}'.format(loss_KL_epoch / no_images)])
    f.close()

