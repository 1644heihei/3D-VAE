import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import model_diffusion

# --- Settings ---
latent_dim = 1024
hidden_dim = 2048
num_layers = 6
batch_size = 64  # Latent vectors are small, so large batch size is fine
lr = 1e-4
epochs = 100
num_timesteps = 1000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Paths
latents_dir = "dataset/latents"
checkpoint_dir = "./checkpoint/diffusion/"
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = "train_diffusion_log.txt"


class LatentDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.lr_dir = os.path.join(root_dir, mode, "LR")
        self.hr_dir = os.path.join(root_dir, mode, "HR")

        if not os.path.exists(self.lr_dir):
            raise FileNotFoundError(
                f"Directory not found: {self.lr_dir}. Please run prepare_latents.py first."
            )

        self.filenames = [f for f in os.listdir(self.lr_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # Load tensors (they were saved as CPU tensors)
        lr_latent = torch.load(os.path.join(self.lr_dir, filename))
        hr_latent = torch.load(os.path.join(self.hr_dir, filename))
        return lr_latent, hr_latent


def train():
    # 1. Initialize Model & Manager
    model = model_diffusion.DiffusionNetwork(
        latent_dim=latent_dim, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)

    manager = model_diffusion.DiffusionManager(
        num_timesteps=num_timesteps, device=device
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 2. Data Loader
    train_dataset = LatentDataset(latents_dir, mode="train")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    print(f"Training Diffusion Model on {len(train_dataset)} latent pairs.")

    # 3. Training Loop
    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for lr_latent, hr_latent in progress_bar:
            lr_latent = lr_latent.to(device)
            hr_latent = hr_latent.to(device)

            # Sample random timesteps
            t = torch.randint(
                0, num_timesteps, (lr_latent.shape[0],), device=device
            ).long()

            # Add noise to HR latent (Forward Process)
            # We want to learn to denoise HR latent, conditioned on LR latent
            x_t, noise = manager.add_noise(hr_latent, t)

            # Predict noise
            predicted_noise = model(x_t, t, lr_latent)

            # Loss
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            progress_bar.set_postfix({"Loss": loss.item()})

            if step % 500 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Epoch {epoch+1}, Step {step}: Loss={loss.item():.6f}\n")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss={avg_loss:.6f}")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(checkpoint_dir, f"diffusion_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    train()
