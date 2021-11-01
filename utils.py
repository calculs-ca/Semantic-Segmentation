import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted

def load_images(path):
    images = []
    img_list = natsorted(os.listdir(path))
    for filename in img_list:
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images

def imshow(img):
    img = torch.squeeze(img)

    imshape = list(img.shape)
    print('image shape:', imshape)
    if imshape[0] == 3:     # RGB image
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img.numpy())
    elif len(imshape) == 2:   # grayscale image
        img = torch.squeeze(img)
        plt.imshow(img.numpy(), cmap='gray', vmin=0, vmax=255)
    plt.show()

# Input image transforms
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
# Mask transform: rgb labels to int
def listToString(arr):
    s = ''.join([str(item) for item in arr])
    return s

# BW-000, HD-001, PF-010, WR-011, RO-100, RI-101, FV-110, SR-111
str2int = { '000': 0, '001': 1, '010': 2, '011': 3, '100':4, '101':5, '110': 6, '111': 7}

def label_mask(mask):
    h, w, ch = mask.shape
    m = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):
            arr = [0 if item == 0 else 1 for item in mask[i][j]]
            m[i][j] = str2int[listToString(arr)]
    return m

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks, transform=transform):
        toTensor = transforms.ToTensor()
        self.images = [transform(img) for img in images]
        self.masks = [toTensor(label_mask(mask)) for mask in masks]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]
