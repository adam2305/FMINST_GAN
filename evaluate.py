import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from model import Generator
from tqdm import tqdm
import os

# Configuration
latent_dim = 100
batch_size = 64
num_classes = 10
generator_epoch = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
val_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=False, transform=transform, download=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize generator
generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)

# Load state dict and remove 'module' prefix if necessary
state_dict = torch.load(f'checkpoints/generator_epoch_{generator_epoch}.pth', map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
generator.load_state_dict(new_state_dict)
generator.eval()

# Create directories to save images
os.makedirs('real_images', exist_ok=True)
os.makedirs('fake_images', exist_ok=True)

# Generate images and save them to disk
with torch.no_grad():
    for i, (real_images, labels) in enumerate(tqdm(val_loader)):
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Generate fake images
        noise = torch.randn(real_images.size(0), latent_dim).to(device)  # Ensure noise has the same batch size as real_images
        fake_images = generator(noise, labels)

        # Save real and fake images
        for j in range(real_images.size(0)):
            real_img = transforms.ToPILImage()(real_images[j].cpu())
            fake_img = transforms.ToPILImage()(fake_images[j].cpu())
            real_img.save(f'real_images/real_{i * batch_size + j}.png')
            fake_img.save(f'fake_images/fake_{i * batch_size + j}.png')

# Calculate FID score
fid_value = fid_score.calculate_fid_given_paths(['real_images', 'fake_images'], batch_size, device, 2048)
print(f"FID score: {fid_value}")