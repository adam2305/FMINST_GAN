import torch
from model import Generator
import matplotlib.pyplot as plt
import numpy as np

latent_dim = 100
num_classes = 10
samples_to_generate = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_images_from_checkpoint(checkpoint_path, num_samples, latent_dim, num_classes, device):
    generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    generator.load_state_dict(new_state_dict)
    generator.eval()
    noise = torch.randn(num_samples, latent_dim).to(device)
    labels = torch.randint(0, num_classes, (num_samples,)).to(device)
    with torch.no_grad():
        generated_images = generator(noise, labels)

    return generated_images, labels

def create_histogram(labels, num_classes, epoch, ax):
    class_counts = np.bincount(labels.cpu().numpy(), minlength=num_classes)
    ax.bar(range(num_classes), class_counts, tick_label=range(num_classes), alpha=0.5)
    ax.set_title(f'Epoch {epoch}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')

epochs = [10, 50, 100, 150, 200]
fig, axes = plt.subplots(1, len(epochs), figsize=(20, 5), sharey=True)

for ax, epoch in zip(axes, epochs):
    checkpoint_path = f'checkpoints/generator_epoch_{epoch}.pth'
    _, labels = generate_images_from_checkpoint(checkpoint_path, samples_to_generate, latent_dim, num_classes, device)
    create_histogram(labels, num_classes, epoch, ax)

plt.suptitle('Class Distribution of Generated Images')
plt.tight_layout()
plt.show()