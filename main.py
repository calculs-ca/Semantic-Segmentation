import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_images, imshow, imgDataset
from models import ConvNet
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
Using 50
"""
# Load images from folder
images = load_images('data/images')
masks = load_images('data/masks')
# Make dataset and apply transforms
img_data = imgDataset(images, masks)
loader = DataLoader(img_data, batch_size=1, shuffle=False)
# Show image sample
img, mask = next(iter(loader))
imshow(mask)

net = ConvNet()
print(net)
print('input size:', img.size())
out = net(img)
print('output shape:', out.size())

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

epochs = 0
for epochs in range(epochs):
    running_loss = 0.

    for image, mask in loader:
        optimizer.zero_grad()

        output = net(image)
        print('output shape:', output.size())
        print('mask shape:', mask.shape)
        print(output)
        print(torch.argmax(output, dim=1))
        #loss = criterion(output, mask)
        # backward
        # step
    # running_loss