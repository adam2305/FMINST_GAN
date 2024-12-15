import torch
import os
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from model import Generator as GenModel, Discriminator as DiscModel

print('Dataset loading...')

latent_dim = 100
batch_size = 64
image_size = 28 * 28
num_classes = 10

data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_data = datasets.FashionMNIST(root='data/FashionMNIST/', train=True, transform=data_transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
print('Dataset Loaded.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 200
learning_rate = 0.0001
disc_train_steps = 5

# Loss function
loss_function = nn.BCELoss()

# Initialize generator and discriminator
gen_model = nn.DataParallel(GenModel(img_dim=latent_dim, class_label_size=num_classes, Image_size=image_size).to(device))
disc_model = nn.DataParallel(DiscModel(class_label_size=num_classes, Image_size=image_size).to(device))

# Optimizers
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=learning_rate)
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=learning_rate)


def train_discriminator(real_imgs, real_labels):
    disc_optimizer.zero_grad()
    batch_size = real_imgs.size(0)
    # Train with real images
    real_validity = disc_model(real_imgs, real_labels)
    real_loss = loss_function(real_validity, torch.ones(batch_size).to(device))
    # Train with fake images
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size)).to(device)
    fake_imgs = gen_model(noise, fake_labels)
    fake_validity = disc_model(fake_imgs, fake_labels)
    fake_loss = loss_function(fake_validity, torch.zeros(batch_size).to(device))

    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    disc_optimizer.step()
    return disc_loss.item()

def train_generator():
    gen_optimizer.zero_grad()
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size)).to(device)
    fake_imgs = gen_model(noise, fake_labels)
    validity = disc_model(fake_imgs, fake_labels)
    gen_loss = loss_function(validity, torch.ones(batch_size).to(device))
    gen_loss.backward()
    gen_optimizer.step()
    return gen_loss.item()

def train():
    disc_loss_list = []
    gen_loss_list = []
    for epoch in tqdm(range(num_epochs)):
        for i, (imgs, labels) in enumerate(train_loader):
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)
            gen_model.train()
            disc_loss = 0
            for _ in range(disc_train_steps):
                disc_loss = train_discriminator(real_imgs, real_labels)
            gen_loss = train_generator()

        print(f"EPOCH: {epoch} | D_Loss: {disc_loss:.5f} | G_Loss: {gen_loss:.5f}")
        disc_loss_list.append(disc_loss)
        gen_loss_list.append(gen_loss)

        if (epoch + 1) % 10 == 0:
            # Save models
            torch.save(gen_model.state_dict(), f'checkpoints/generator_epoch_{epoch + 1}.pth')
            torch.save(disc_model.state_dict(), f'checkpoints/discriminator_epoch_{epoch + 1}.pth')

            # Save losses
            torch.save({'G_losses': gen_loss_list, 'D_losses': disc_loss_list}, f'history/losses_epoch_{epoch + 1}.pth')

    # Create directories if they do not exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('history', exist_ok=True)

    # Save models
    torch.save(gen_model.state_dict(), 'checkpoints/generator.pth')
    torch.save(disc_model.state_dict(), 'checkpoints/discriminator.pth')

    # Save losses
    torch.save({'G_losses': gen_loss_list, 'D_losses': disc_loss_list}, 'history/losses.pth')

if __name__ == '__main__':
    train()