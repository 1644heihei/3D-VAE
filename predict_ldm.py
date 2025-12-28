import torch
import os
import numpy as np
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import model_vae_3d as model_vae
import model_diffusion
import dataloader_pair

# --- Settings ---
latent_dim = 1024
hidden_dim = 2048
num_layers = 6
num_timesteps = 1000
ddim_steps = 50  # DDIM Sampling steps (Faster than 1000)
batch_size = 1  # Process one by one for analysis
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Paths
vae_checkpoint_dir = "./checkpoint/vae_denoise/"
diffusion_checkpoint_dir = "./checkpoint/diffusion/"
output_dir = "./results/ldm_prediction"
os.makedirs(output_dir, exist_ok=True)


def get_latest_checkpoint(dir_path):
    list_of_files = glob.glob(os.path.join(dir_path, "*.pt"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def calculate_psnr(img1, img2, data_range=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))


def save_slice(vol, fname, normalize=True):
    # Save the middle slice of the volume
    # vol shape: (C, D, H, W) -> (1, 80, 96, 80)
    # Slice index: D//2
    d = vol.shape[2]  # D dimension is at index 2
    slice_img = vol[0, 0, d // 2, :, :].cpu().numpy()

    if normalize:
        # Normalize to 0-255
        slice_img = (slice_img - slice_img.min()) / (
            slice_img.max() - slice_img.min() + 1e-8
        )
        slice_img = (slice_img * 255).astype(np.uint8)

    img = Image.fromarray(slice_img)
    img.save(fname)


def predict():
    # 1. Load Models
    # VAE
    vae_path = get_latest_checkpoint(vae_checkpoint_dir)
    if not vae_path:
        print("VAE checkpoint not found. Please run Phase 1 (train_vae_pair.py) first.")
        return
    print(f"Loading VAE: {vae_path}")
    vae = model_vae.VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()

    # Diffusion
    diff_path = get_latest_checkpoint(diffusion_checkpoint_dir)
    if not diff_path:
        print(
            "Diffusion checkpoint not found. Please run Phase 3 (train_diffusion.py) first."
        )
        return
    print(f"Loading Diffusion: {diff_path}")
    diffusion = model_diffusion.DiffusionNetwork(
        latent_dim=latent_dim, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)
    diffusion.load_state_dict(torch.load(diff_path, map_location=device))
    diffusion.eval()

    manager = model_diffusion.DiffusionManager(
        num_timesteps=num_timesteps, device=device
    )

    # 2. Load Test Data
    try:
        dataset = dataloader_pair.PairedMRIDataset(root_dir="dataset", mode="test")
    except FileNotFoundError:
        print("Test dataset not found. Please run prepare_data.py first.")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Processing {len(dataset)} test samples...")
    print(f"Results will be saved to: {output_dir}")

    total_psnr_baseline = 0
    total_psnr_result = 0
    count = 0

    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(tqdm(dataloader)):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # --- Step 1: Encode LR to Latent ---
            # VAE Encoder -> Flatten -> z_mean
            x_lr = vae.encoder(lr_img)
            x_lr = torch.flatten(x_lr, start_dim=1)
            z_lr = vae.z_mean(x_lr)  # Use mean for condition

            # --- Step 2: Diffusion Sampling (DDIM) ---
            # Generate z_restored from noise, conditioned on z_lr
            # This is the core of LDM: Denoising in latent space
            z_restored = manager.sample_ddim(
                diffusion, z_lr, num_inference_steps=ddim_steps
            )

            # --- Step 3: Decode to Image ---
            pred_img = vae.decoder(z_restored)

            # --- Evaluation ---
            # Baseline: LR vs HR
            psnr_base = calculate_psnr(lr_img, hr_img)
            # Result: Pred vs HR
            psnr_res = calculate_psnr(pred_img, hr_img)

            total_psnr_baseline += psnr_base.item()
            total_psnr_result += psnr_res.item()
            count += 1

            # --- Save Results (First 20 samples) ---
            if i < 20:
                save_base = os.path.join(output_dir, f"sample_{i:03d}")
                os.makedirs(save_base, exist_ok=True)

                # Save slices (Middle slice of Depth)
                save_slice(lr_img, os.path.join(save_base, "1_input_LR.png"))
                save_slice(hr_img, os.path.join(save_base, "2_target_HR.png"))
                save_slice(pred_img, os.path.join(save_base, "3_output_LDM.png"))

                # Save text info
                with open(os.path.join(save_base, "metrics.txt"), "w") as f:
                    f.write(f"Baseline PSNR: {psnr_base.item():.4f} dB\n")
                    f.write(f"Result PSNR:   {psnr_res.item():.4f} dB\n")

    print("\n=== Final Results ===")
    print(f"Average Baseline PSNR: {total_psnr_baseline/count:.4f} dB")
    print(f"Average Result PSNR:   {total_psnr_result/count:.4f} dB")
    print(
        f"Improvement:           {(total_psnr_result - total_psnr_baseline)/count:+.4f} dB"
    )


if __name__ == "__main__":
    predict()
