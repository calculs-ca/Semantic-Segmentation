import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import load_images
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
Using 50
"""
images = load_images('data/images')
masks = load_images('data/masks')

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]

img_data = imgDataset(images, masks)
loader = DataLoader(img_data, batch_size=1, shuffle=False)
img, mask = next(iter(loader))
img = torch.squeeze(img)

fig = plt.figure()
plt.imshow(img.numpy())
plt.show()