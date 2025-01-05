import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from model import Generator

latent_dim = 100
num_classes = 10
samples_to_generate = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_generator(checkpoint_path, latent_dim, num_classes, device):
    generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    generator.load_state_dict(new_state_dict)
    generator.eval()
    return generator

def generate_latent_vectors(generator, num_samples, latent_dim, num_classes, device):
    noise = torch.randn(num_samples, latent_dim).to(device)
    labels = torch.randint(0, num_classes, (num_samples,)).to(device)
    return noise, labels

def visualize_latent_space_tsne(latent_vectors, labels):
    tsne = TSNE(n_components=3, random_state=42)
    latent_3d = tsne.fit_transform(latent_vectors.cpu().numpy())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], c=labels.cpu().numpy(), cmap='tab10', alpha=0.7)
    fig.colorbar(scatter, ticks=range(num_classes))
    ax.set_title('t-SNE visualization of latent space')
    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')
    ax.set_zlabel('t-SNE component 3')
    plt.show()

def visualize_latent_space_kmeans(latent_vectors, labels, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_vectors.cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0].cpu().numpy(), latent_vectors[:, 1].cpu().numpy(), c=cluster_labels,
                          cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_clusters))
    plt.title('K-means clustering of latent space')
    plt.xlabel('Latent dimension 1')
    plt.ylabel('Latent dimension 2')
    plt.show()


checkpoint_path = 'checkpoints/generator_epoch_200.pth'
generator = load_generator(checkpoint_path, latent_dim, num_classes, device)
latent_vectors, labels = generate_latent_vectors(generator, samples_to_generate, latent_dim, num_classes, device)
visualize_latent_space_tsne(latent_vectors, labels)
visualize_latent_space_kmeans(latent_vectors, labels, num_classes)