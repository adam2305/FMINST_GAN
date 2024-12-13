import torch
import torchvision
import os
import argparse

from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Conditional GAN samples.')
    parser.add_argument("--batch_size", type=int, default=2048, help="The batch size to use for generation.")
    parser.add_argument("--n_samples", type=int, default=10000, help="The number of samples to generate.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator(g_output_dim=mnist_dim, num_classes=num_classes).to(device)
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < args.n_samples:
            z = torch.randn(args.batch_size, 100).to(device)
            gen_labels = torch.randint(0, num_classes, (args.batch_size,)).to(device)
            x = model(z, gen_labels)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < args.n_samples:
                    torchvision.utils.save_image(x[k:k + 1], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1

    print('Generation done')