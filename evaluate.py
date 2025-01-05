import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from model import Generator
from tqdm import tqdm
import os


latent_dim = 100
batch_size = 64
num_classes = 10
generator_epoch = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
val_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=False, transform=transform, download=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)

state_dict = torch.load(f'checkpoints/generator_epoch_{generator_epoch}.pth', map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
generator.load_state_dict(new_state_dict)
generator.eval()

os.makedirs('real_images', exist_ok=True)
os.makedirs('fake_images', exist_ok=True)

with torch.no_grad():
    for i, (real_images, labels) in enumerate(tqdm(val_loader)):
        real_images = real_images.to(device)
        labels = labels.to(device)
        noise = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(noise, labels)
        for j in range(real_images.size(0)):
            real_img = transforms.ToPILImage()(real_images[j].to(device))
            fake_img = transforms.ToPILImage()(fake_images[j].to(device))
            real_img.save(f'real_images/real_{i * batch_size + j}.png')
            fake_img.save(f'fake_images/fake_{i * batch_size + j}.png')

fid_value = fid_score.calculate_fid_given_paths(['real_images', 'fake_images'], batch_size, device, 2048)
print(f"FID score: {fid_value}")