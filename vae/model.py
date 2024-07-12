import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 48 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (48, 16, 16))
        self.conv1 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.sigmoid(self.conv3(x))
        return x


class VAE(nn.Module):

    def __init__(self, latent_dim, device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')):
        super().__init__()
        self.latent_dim = latent_dim
        # the encoder generates mean vector & standard deviation vector, each has the dimension of self.latent_dim
        self.encoder_dim = 2 * latent_dim
        self.encoder = Encoder(self.encoder_dim)
        self.decoder = Decoder(latent_dim)
        self.device = device
        self.to(device)

    def reparameterize(self, z_mean, z_logsigma):
        # By default, torch.randn is "standard" (ie. mean=0 and std=1.0)
        epsilon = torch.randn(z_mean.shape, device=self.device)
        z = z_mean + torch.exp(0.5 * z_logsigma) * epsilon
        return z

    def encode(self, x):
        out = self.encoder(x)
        z_mean = out[:, :self.latent_dim]
        z_logsigma = out[:, self.latent_dim:]
        return z_mean, z_logsigma

    def decode(self, x):
        reconstruction = self.decoder(x)
        return reconstruction

    def forward(self, x):
        z_mean, z_logsigma = self.encode(x)
        z = self.reparameterize(z_mean, z_logsigma)
        recon = self.decode(z)
        return x, recon, z_mean, z_logsigma


if __name__ == '__main__':
    import os
    from PIL import Image
    from torch.utils.data import DataLoader
    from dataset import TinyImageNet, CelebADataset
    # I downloaded the datasets preemptively
    base_data_dir = "/Users/bittergreen/datasets/vision/"  # dataset directory
    imagenet_data_dir = os.path.join(base_data_dir, "tiny-imagenet-200")
    celeba_data_dir = os.path.join(base_data_dir, "celeba")

    imagenet = TinyImageNet(imagenet_data_dir)
    celeba = CelebADataset(celeba_data_dir)

    # Initialize the dataloader
    imagenet_loader = DataLoader(imagenet, batch_size=32, shuffle=True)
    celeba_loader = DataLoader(celeba, batch_size=32, shuffle=True)

    dim = 12
    encoder = Encoder(dim)
    decoder = Decoder(dim)
    # Iterate over the data
    for images in celeba_loader:
        z = encoder(images)
        org = decoder(z)
        img = org[0].detach().permute(1, 2, 0).numpy()
        img = Image.fromarray(img.astype('uint8'))
        img.show()
        break
