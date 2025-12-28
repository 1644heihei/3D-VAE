"""
Train VAE for Image Quality Improvement (Supervised)
Input: Low Quality (LR) -> VAE -> Output
Target: High Quality (HR)
Loss: Reconstruction(Output, HR) + KL(z)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import model_vae_3d as model_vae  # Renamed from model3
import dataloader_pair  # The new paired dataloader
import loss

##---------Settings--------------------------
batch_size = 2  # Reduced to avoid OOM
lrate = 0.001
epochs = 50
weight_decay = 5e-7
kl_weight = 0.01  # Weight for KL divergence
latent_dim = 1024  # Reduced from 16384 to avoid OOM

# Paths
train_dir = "dataset/train"  # Contains LR and HR folders
checkpoint_dir = "./checkpoint/vae_denoise/"
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = "train_vae_denoise_log.txt"

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


def train():
    # Model
    # latent_dim reduced to 1024 to fit in 24GB VRAM
    vae_model = model_vae.VAE(latent_dim=latent_dim)
    vae_model.to(device)

    # Optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=lrate, weight_decay=weight_decay)

    # Loss functions
    criterion_rec = loss.L1Loss()  # Use custom L1Loss from loss.py
    criterion_kl = loss.KLDivergence()

    # Dataset & Dataloader
    # Assuming dataloader_pair has PairedMRIDataset
    dataset = dataloader_pair.PairedMRIDataset(root_dir="dataset", mode="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Number of training pairs: {len(dataset)}")

    step = 0

    for epoch in range(epochs):
        vae_model.train()
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            # VAE returns: y, z_mean, z_log_sigma
            outputs, z_mean, z_log_sigma = vae_model(inputs)

            # Calculate Loss
            # Reconstruction Loss: Compare Output with Target (HR)
            loss_rec = criterion_rec(outputs, targets)

            # KL Divergence Loss
            loss_kl = criterion_kl(z_mean, z_log_sigma)

            # Total Loss
            total_loss = loss_rec + kl_weight * loss_kl

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            epoch_rec_loss += loss_rec.item()
            epoch_kl_loss += loss_kl.item()
            epoch_total_loss += total_loss.item()

            step += 1
            progress_bar.set_postfix(
                {
                    "Rec": loss_rec.item(),
                    "KL": loss_kl.item(),
                    "Total": total_loss.item(),
                }
            )

            if step % 100 == 0:
                with open(log_file, "a") as f:
                    f.write(
                        f"Epoch {epoch+1}, Step {step}: Rec={loss_rec.item():.6f}, KL={loss_kl.item():.6f}, Total={total_loss.item():.6f}\n"
                    )

        # Epoch Summary
        avg_rec = epoch_rec_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)
        avg_total = epoch_total_loss / len(dataloader)

        print(
            f"Epoch {epoch+1} Summary: Rec={avg_rec:.6f}, KL={avg_kl:.6f}, Total={avg_total:.6f}"
        )

        # Save Checkpoint
        save_path = os.path.join(checkpoint_dir, f"vae_denoise_epoch_{epoch+1}.pt")
        torch.save(vae_model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    train()
