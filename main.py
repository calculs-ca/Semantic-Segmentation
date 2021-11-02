import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_images, imshow, imgDataset
from models import ConvNet, UNet
"""
Dataset: Underwater imagery (SUIM)
Using 50
"""
# Select model: 'unet', 'conv'
model = 'unet'

# Load images from folder
images = load_images('data/images')
masks = load_images('data/masks')
# Make dataset and apply transforms
img_data = imgDataset(images, masks)
loader = DataLoader(img_data, batch_size=1, shuffle=False)
# Show image sample
img, mask = next(iter(loader))
#imshow(mask)

if model == 'unet':
    net = UNet()
else:
    net = ConvNet()
print(net)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

epochs = 3
for epochs in range(epochs):
    running_loss = 0.

    for image, mask in loader:
        mask = torch.squeeze(mask, dim=1)
        optimizer.zero_grad()

        output = net(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('loss:', running_loss)