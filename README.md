# FMINST_GAN

## Overview
FMINST_GAN is a project that implements a Generative Adversarial Network (GAN) to generate images of clothing items from the Fashion MNIST dataset. The project includes training the GAN, visualizing the generated images, and analyzing the latent space.

## Project Structure
- `train.py`: Script to train the GAN model.
- `plot_loss.py`: Script to plot the generator and discriminator losses.
- `plot_samples.py`: Script to generate and save sample images from the trained generator.
- `model.py`: Contains the definitions of the Generator and Discriminator models.
- `checkpoints/`: Directory to save model checkpoints.
- `history/`: Directory to save training history (losses).
- `images/`: Directory to save generated images.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- scikit-learn

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/FMINST_GAN.git
    cd FMINST_GAN
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the GAN
To train the GAN, run:
```sh
python train.py