import torch
import matplotlib.pyplot as plt

epoch_number = 200

# Load the losses
losses = torch.load(f'history/losses_epoch_{epoch_number}.pth')

# Load the losses
losses = torch.load('history/losses.pth')
G_losses = losses['G_losses']
D_losses = losses['D_losses']

print("index of min", G_losses.index(min(G_losses)))

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Generator and Discriminator Losses')
plt.show()