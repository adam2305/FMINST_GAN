import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator
from utils import D_train, G_train, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Conditional GAN.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD")

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='data/FashionMNIST/', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    print('Model Loading...')
    mnist_dim = 784
    num_classes = 10
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim, num_classes=num_classes)).to(device)
    D = torch.nn.DataParallel(Discriminator(d_input_dim=mnist_dim, num_classes=num_classes)).to(device)

    print('Model loaded.')
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    print('Start Training :')

    n_epoch = args.epochs
    for epoch in trange(1, n_epoch + 1, leave=True):
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            labels = labels.to(device)
            D_train(x, labels, G, D, D_optimizer, criterion, device)
            G_train(x, labels, G, D, G_optimizer, criterion, device)


    save_models(G, D, 'checkpoints')

    print('Training done')