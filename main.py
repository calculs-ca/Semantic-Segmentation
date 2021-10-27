import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import load_images, imshow
from models import ConvNet
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
Using 50
"""
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        if transform is not None:
            self.images = [transform(img) for img in images]
            self.masks = [transform(mask) for mask in masks]
        else:
            self.images = images
            self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]

# Load images from folder
images = load_images('data/images')
masks = load_images('data/masks')
# Make dataset and apply transforms
img_data = imgDataset(images, masks, transform=transform)
loader = DataLoader(img_data, batch_size=1, shuffle=False)
# Show image sample
img, mask = next(iter(loader))
#imshow(img)

net = ConvNet()
print(net)
print('input size:', img.size())
out = net(img)
print('output shape:', out.size())