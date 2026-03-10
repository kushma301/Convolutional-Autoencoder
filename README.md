# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS

STEP 1:
Import the required libraries including PyTorch, torchvision, and matplotlib.

STEP 2:
Load the MNIST dataset with proper transforms (e.g., transforms.ToTensor()).

STEP 3:
Define a function to add Gaussian noise to the dataset.

STEP 4:
Design the convolutional autoencoder architecture with encoder and decoder blocks using nn.Conv2d and nn.ConvTranspose2d.

STEP 5:
Initialize the model, define the loss function (MSELoss), and select an optimizer (Adam).

STEP 6:
Train the model using noisy images as input and clean images as targets.

STEP 7:
Visualize the performance by plotting original, noisy, and reconstructed images.

## PROGRAM
### Name:  KUSHMA S
### Register Number:  212224040168

```
# Autoencoder for Image Denoising using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2), 
            nn.Sigmoid()  # Ensure output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")


# Evaluate and visualize
def visualize_denoising(model, loader, num_images=5):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:  KUSHMA S")
    print("Register Number: 212224040168")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

## OUTPUT

### Model Summary

<img width="953" height="575" alt="image" src="https://github.com/user-attachments/assets/346de768-71e6-42c0-9afe-e1f0f2b1a7f2" />

### Original vs Noisy Vs Reconstructed Image

<img width="1063" height="572" alt="image" src="https://github.com/user-attachments/assets/af372e18-b489-4010-9fb7-82f385079e0b" />


## RESULT
A convolutional autoencoder was successfully implemented using PyTorch to denoise MNIST images. The model was able to reconstruct clean images from noisy inputs, demonstrating its effectiveness in feature learning and denoising tasks.
