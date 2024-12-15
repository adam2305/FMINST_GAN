import torch
from model import Generator
from matplotlib import pyplot as plt

epoch_number = 100

img_dim = 100
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)

# Load the state dictionary and remove 'module.' prefix
state_dict = torch.load(f'checkpoints/generator_epoch_{epoch_number}.pth', map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
generator.load_state_dict(new_state_dict)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

noise_test = torch.randn(img_dim)
noise_test.unsqueeze_(dim=0)
label_list = []
for x in range(0, 10):
    label_list.append(torch.LongTensor([x]))

generator.eval()
gan_img_test_list = []
for x in range(0, 10):
    with torch.inference_mode():
        gan_img_test = generator(noise_test, label_list[x])
        gan_img_test = gan_img_test.squeeze().reshape(28, 28).to(device).detach().numpy()
        gan_img_test_list.append(gan_img_test)

f = plt.figure(figsize=(16, 16))
for x in range(0, 10):
    f.add_subplot(1, 10, x + 1)
    plt.imshow(gan_img_test_list[x], cmap='gray')
    plt.title(f"{class_names[x]}")
plt.show()