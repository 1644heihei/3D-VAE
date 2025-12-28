import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import model_vae_3d as model_vae
import dataloader_pair
from tqdm import tqdm

# --- Settings ---
latent_dim = 1024  # Must match training
batch_size = 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Path to the latest checkpoint (or specify a file path directly)
checkpoint_dir = "./checkpoint/vae_denoise/"
# Find the latest checkpoint file
list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
if not list_of_files:
    print("No checkpoints found. Please wait for training to save a model.")
    exit()
latest_file = max(list_of_files, key=os.path.getctime)
checkpoint_path = latest_file
# checkpoint_path = "./checkpoint/vae_denoise/vae_denoise_epoch_10.pt" # Or specify manually

print(f"Evaluating checkpoint: {checkpoint_path}")


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate PSNR between two tensors.
    img1, img2: tensors of shape (N, C, D, H, W) or (N, C, H, W)
    data_range: The dynamic range of the images (1.0 for 0-1 normalized data)
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3, 4])
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr


def evaluate():
    # Load Model
    model = model_vae.VAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Data
    # Use 'test' set for evaluation
    try:
        dataset = dataloader_pair.PairedMRIDataset(root_dir="dataset", mode="test")
    except FileNotFoundError:
        print("Test dataset not found. Please run prepare_data.py first.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    total_psnr_baseline = 0
    total_psnr_result = 0
    count = 0

    print("Starting evaluation...")

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs, _, _ = model(inputs)

            # Calculate PSNR
            # 1. Baseline: Input (Low Quality) vs Target (High Quality)
            psnr_baseline = calculate_psnr(inputs, targets)

            # 2. Result: Output (Restored) vs Target (High Quality)
            psnr_result = calculate_psnr(outputs, targets)

            total_psnr_baseline += psnr_baseline.sum().item()
            total_psnr_result += psnr_result.sum().item()
            count += inputs.size(0)

    avg_psnr_baseline = total_psnr_baseline / count
    avg_psnr_result = total_psnr_result / count

    print("\n=== Evaluation Results ===")
    print(f"Number of samples: {count}")
    print(f"Baseline PSNR (Input vs Target): {avg_psnr_baseline:.4f} dB")
    print(f"Result PSNR   (Output vs Target): {avg_psnr_result:.4f} dB")

    improvement = avg_psnr_result - avg_psnr_baseline
    print(f"Improvement: {improvement:+.4f} dB")

    if improvement > 0:
        print("SUCCESS: Image quality has improved.")
    else:
        print("WARNING: Image quality has degraded (or model hasn't learned yet).")


if __name__ == "__main__":
    evaluate()
