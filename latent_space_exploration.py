import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import Generator

# Configuration
latent_dim = 100
num_classes = 10
samples_to_generate = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the pretrained model
def load_generator(checkpoint_path, latent_dim, num_classes, device):
    generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    generator.load_state_dict(new_state_dict)
    generator.eval()
    return generator

# Function to generate latent vectors and labels
def generate_latent_vectors(generator, num_samples, latent_dim, num_classes, device):
    noise = torch.randn(num_samples, latent_dim).to(device)
    labels = torch.randint(0, num_classes, (num_samples,)).to(device)
    return noise, labels

# Function to visualize latent space using t-SNE
def visualize_latent_space_tsne(latent_vectors, labels, initial_dim=100):
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    latent_2d = tsne.fit_transform(latent_vectors[:, :initial_dim].cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes))
    plt.title('t-SNE visualization of latent space')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()

# Load the generator model
checkpoint_path = 'checkpoints/generator_epoch_200.pth'
generator = load_generator(checkpoint_path, latent_dim, num_classes, device)

# Generate latent vectors and labels
latent_vectors, labels = generate_latent_vectors(generator, samples_to_generate, latent_dim, num_classes, device)

# Visualize latent space using t-SNE
visualize_latent_space_tsne(latent_vectors, labels)