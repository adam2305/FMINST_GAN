import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
from model import Generator
import numpy as np

# Configuration
batch_size = 64
latent_dim = 100
num_classes = 10
samples_per_class = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Inception v3 model
inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

# Function to calculate Inception Score
def calculate_inception_score(images, inception_model, splits=10):
    N = len(images)
    dataloader = DataLoader(images, batch_size=batch_size)
    preds = np.zeros((N, 1000))

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            batch = batch.to(device)
            batch_size_i = batch.size(0)
            pred = inception_model(batch)[0]
            preds[i * batch_size:i * batch_size + batch_size_i] = pred.cpu().numpy()

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# Load Fashion MNIST test dataset
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to 299x299 for Inception v3
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

test_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Generate samples
def generate_samples(generator, latent_dim, num_classes, samples_per_class, device):
    generator.eval()
    generated_images = []

    with torch.no_grad():
        for class_label in range(num_classes):
            noise = torch.randn(samples_per_class, latent_dim).to(device)
            labels = torch.full((samples_per_class,), class_label, dtype=torch.long).to(device)
            fake_images = generator(noise, labels)
            generated_images.append(fake_images)

    generated_images = torch.cat(generated_images)
    return generated_images

# Load the generator model
generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)
state_dict = torch.load('checkpoints/generator_epoch_200.pth', map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
generator.load_state_dict(new_state_dict)

# Generate images
generated_images = generate_samples(generator, latent_dim, num_classes, samples_per_class, device)

# Apply transform to each generated image
generated_images = torch.stack([transform(img) for img in generated_images])

# Calculate Inception Score for validation dataset
validation_images = torch.stack([data[0] for data in test_dataset])
validation_score, validation_std = calculate_inception_score(validation_images, inception_model)
print(f'Validation Inception Score: {validation_score:.4f} ± {validation_std:.4f}')

# Calculate Inception Score for generated dataset
generated_score, generated_std = calculate_inception_score(generated_images, inception_model)
print(f'Generated Inception Score: {generated_score:.4f} ± {generated_std:.4f}')