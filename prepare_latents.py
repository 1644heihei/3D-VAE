import torch
import os
import glob
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import model_vae_3d as model_vae
import dataloader_pair

# --- Settings ---
latent_dim = 1024  # Phase 1で設定した値と同じにする
batch_size = 16  # 画像処理がないので少し大きくてもOK
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Paths
checkpoint_dir = "./checkpoint/vae_denoise/"
output_dir = "dataset/latents"
os.makedirs(os.path.join(output_dir, "LR"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "HR"), exist_ok=True)


def get_latest_checkpoint(dir_path):
    list_of_files = glob.glob(os.path.join(dir_path, "*.pt"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def encode_and_save_split(mode):
    print(f"Processing {mode} split...")
    output_subdir = os.path.join(output_dir, mode)
    os.makedirs(os.path.join(output_subdir, "LR"), exist_ok=True)
    os.makedirs(os.path.join(output_subdir, "HR"), exist_ok=True)

    # 2. Prepare Data Loader
    try:
        dataset = dataloader_pair.PairedMRIDataset(root_dir="dataset", mode=mode)
    except FileNotFoundError:
        print(f"Skipping {mode}: Dataset not found.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Processing {len(dataset)} pairs for {mode}...")

    # 3. Processing Loop
    count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"Encoding {mode}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # --- Encode LR images ---
            x_lr = vae.encoder(inputs)
            x_lr = torch.flatten(x_lr, start_dim=1)
            z_mean_lr = vae.z_mean(x_lr)

            # --- Encode HR images ---
            x_hr = vae.encoder(targets)
            x_hr = torch.flatten(x_hr, start_dim=1)
            z_mean_hr = vae.z_mean(x_hr)

            # --- Save to disk ---
            for i in range(inputs.size(0)):
                save_name = f"latent_{count:06d}.pt"
                torch.save(
                    z_mean_lr[i].cpu(), os.path.join(output_subdir, "LR", save_name)
                )
                torch.save(
                    z_mean_hr[i].cpu(), os.path.join(output_subdir, "HR", save_name)
                )
                count += 1

    print(f"Finished {mode}! Saved {count} latent pairs.")


def encode_and_save():
    # 1. Load VAE Model (Global)
    global vae
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print("Error: No checkpoint found in", checkpoint_dir)
        return

    print(f"Loading VAE checkpoint: {checkpoint_path}")
    vae = model_vae.VAE(latent_dim=latent_dim)
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vae.to(device)
    vae.eval()

    # Process all splits
    for mode in ["train", "test"]:
        encode_and_save_split(mode)


if __name__ == "__main__":
    encode_and_save()
