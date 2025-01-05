import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
from model import Generator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuration
batch_size = 64
num_epochs = 10
learning_rate = 0.001
latent_dim = 100
num_classes = 10
samples_per_class = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Fashion MNIST train and test datasets
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28 for the simple CNN
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to generate samples
def generate_samples(generator, latent_dim, num_classes, samples_per_class, device):
    generator.eval()
    generated_images = []
    generated_labels = []

    with torch.no_grad():
        for class_label in range(num_classes):
            noise = torch.randn(samples_per_class, latent_dim).to(device)
            labels = torch.full((samples_per_class,), class_label, dtype=torch.long).to(device)
            fake_images = generator(noise, labels)
            generated_images.append(fake_images)
            generated_labels.append(labels)

    generated_images = torch.cat(generated_images)
    generated_labels = torch.cat(generated_labels)
    return generated_images, generated_labels

# Load the generator model
generator = Generator(img_dim=latent_dim, class_label_size=num_classes, Image_size=28 * 28).to(device)
state_dict = torch.load('checkpoints/generator_epoch_200.pth', map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
generator.load_state_dict(new_state_dict)

# Generate 2000 samples (200 per class)
generated_images, generated_labels = generate_samples(generator, latent_dim, num_classes, samples_per_class, device)
transform = transforms.Compose([transforms.Normalize(mean=(0.5,), std=(0.5,))])
generated_images = transform(generated_images)
generated_dataset = TensorDataset(generated_images.unsqueeze(1), generated_labels)

# Combine train dataset with generated dataset
combined_dataset = ConcatDataset([train_dataset, generated_dataset])
combined_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True)

# Training loop
# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for images, labels in combined_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(combined_loader):.4f}")

# Validation loop
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

# Save confusion matrix for validation samples
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Samples')
plt.savefig('confusion_matrix_validation.png')