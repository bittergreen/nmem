import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TinyImageNet, CelebADataset

"""
Currently the model is not doing well, it tends to generate something more like the average of 
faces. 
"""


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
    # Latent loss (KL Divergence)
    latent_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=1)
    latent_loss = kl_weight * latent_loss.mean()
    # Reconstruction loss (L1 Loss)
    x = x.view(x.size(0), -1)
    x_recon = x_recon.view(x_recon.size(0), -1)
    # Calculate the reconstruction loss
    reconstruction_loss = F.l1_loss(x_recon, x, reduction='mean')
    # Total VAE loss
    vae_loss = latent_loss + reconstruction_loss
    return vae_loss


def show_images(original, reconstructed, epoch, batch):
    original = original.cpu().detach()[0]  # Select the first image in the batch
    reconstructed = reconstructed.cpu().detach()[0]

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed.permute(1, 2, 0))
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.suptitle(f'Epoch {epoch + 1}, Batch {batch + 1}')
    plt.show(block=False)  # Show the images without blocking
    plt.pause(1.0)  # Pause for 1 second
    plt.close()  # Close the figure to free up resources


class VAETrainer:

    def __init__(self, vae, batch_size=32, lr=0.001, num_epochs=1):
        self.vae = vae
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.celeba_loader = None
        self._load_datasets()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # train with M3 if available
        self.vae.to(self.device)

    def _load_datasets(self):
        base_data_dir = "/Users/bittergreen/datasets/vision/"
        celeba_data_dir = os.path.join(base_data_dir, "celeba")
        celeba = CelebADataset(celeba_data_dir)
        self.celeba_loader = DataLoader(celeba, batch_size=self.batch_size, shuffle=True)

    def train(self):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
        for epoch in range(self.num_epochs):
            i = 0
            for batch in tqdm(self.celeba_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", unit="batch"):
                optimizer.zero_grad()
                batch = batch.to(self.device)
                x, recon, z_mean, z_logsigma = self.vae(batch)
                loss = vae_loss_function(x, recon, z_mean, z_logsigma)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    tqdm.write(f"Loss: {loss.item():.4f}")
                    show_images(x, recon, epoch, i)
                i += 1

    def save(self, save_path="./vae.pth"):
        torch.save(self.vae.state_dict(), save_path)


if __name__ == '__main__':
    from model import VAE
    vae = VAE(latent_dim=12)
    trainer = VAETrainer(vae)
    trainer.train()
    trainer.save()

