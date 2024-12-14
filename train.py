import torch
import os
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

print('Dataset loading...')

img_dim = 100
BATCH_SIZE = 64
Image_size = 28 * 28
class_label_size = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print('Dataset Loaded.')

############################################################################################################################################################################

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        self.model = nn.Sequential(
            nn.Linear(img_dim + class_label_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, Image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(class_label_size, class_label_size)

        self.model = nn.Sequential(
            nn.Linear(Image_size + class_label_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 200
learming_rate = 0.0001
Times_train_discrimnizator=5

# Loss function
criterion = nn.BCELoss()

# Initialize generator and discriminator
generator = nn.DataParallel(Generator().to(device))
discriminator = nn.DataParallel(Discriminator().to(device))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learming_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learming_rate)


def train_discriminator(real_images, labels):
    optimizer_D.zero_grad()
    batch_size = real_images.size(0)
    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.ones(batch_size).to(device))
    # train with fake images
    z = torch.randn(batch_size, 100).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()
    return d_loss.item()

def train_generator():
    optimizer_G.zero_grad()
    z =torch.randn(BATCH_SIZE, 100).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, 10, BATCH_SIZE)).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, torch.ones(BATCH_SIZE).to(device))
    g_loss.backward()
    optimizer_G.step()
    return g_loss.item()

def train():
    d_loss_list=[]
    g_loss_list=[]
    for epoch in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            real_images = images.to(device)
            labels = labels.to(device)
            generator.train()
            d_loss = 0
            for _ in range(Times_train_discrimnizator):
                d_loss = train_discriminator(real_images, labels)
            g_loss = train_generator()

        print(f"EPOCH: {epoch} | D_Loss: {d_loss:.5f} | G_Loss: {g_loss:.5f}")
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)

        if (epoch + 1) % 10 == 0:
            # Save models
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch + 1}.pth')

            # Save losses
            torch.save({'G_losses': g_loss_list, 'D_losses': d_loss_list}, f'history/losses_epoch_{epoch + 1}.pth')

    # Create directories if they do not exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('history', exist_ok=True)


    # Save models
    torch.save(generator.state_dict(), 'checkpoints/generator.pth')
    torch.save(discriminator.state_dict(), 'checkpoints/discriminator.pth')

    # Save losses
    torch.save({'G_losses': g_loss_list, 'D_losses': d_loss_list}, 'history/losses.pth')

if __name__ == '__main__':
    train()

