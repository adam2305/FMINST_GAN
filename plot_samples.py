import torch
from model import Generator
from matplotlib import pyplot as plt
import os

img_dim = 100
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_images(epoch_number):
    generator = Generator().to(device)
    state_dict = torch.load(f'checkpoints/generator_epoch_{epoch_number}.pth', map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    generator.load_state_dict(new_state_dict)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    noise_test = torch.randn(img_dim).unsqueeze(0).to(device)
    label_list = [torch.LongTensor([x]).to(device) for x in range(10)]
    generator.eval()
    gan_img_test_list = []
    for label in label_list:
        with torch.inference_mode():
            gan_img_test = generator(noise_test, label)
            gan_img_test = gan_img_test.squeeze().reshape(28, 28).cpu().detach().numpy()
            gan_img_test_list.append(gan_img_test)

    return gan_img_test_list

def save_all_images():
    epochs = [10, 50, 100, 150, 200]
    rows = len(epochs)
    cols = 10
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 * rows / cols))
    for row, epoch in enumerate(epochs):
        gan_img_test_list = generate_images(epoch)
        for i, img in enumerate(gan_img_test_list):
            ax = axes[row, i]
            ax.imshow(img, cmap='gray')
            if i == 0:
                ax.set_ylabel(f"Epoch {epoch}", fontsize=12)
            ax.set_title(class_names[i])
            ax.axis('on')
    plt.tight_layout()
    plt.savefig('images/all_epochs.png')
    plt.close()

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    save_all_images()