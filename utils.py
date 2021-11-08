import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted

def load_images(path):
    images = []
    img_list = natsorted(os.listdir(path))
    for f in img_list:
        img_path = os.path.join(path, f)
        img = np.array(Image.open(img_path).convert('RGB'))
        if img is not None:
            images.append(img)
    return images

def imshow(img):
    img = torch.squeeze(img)

    imshape = list(img.shape)
    if imshape[0] == 3:         # RGB image
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img.numpy())
    elif len(imshape) == 2:     # grayscale image
        img = torch.squeeze(img)
        n_classes = 7
        plt.imshow(img.numpy(), cmap='gray', vmin=0, vmax=n_classes)

# Show input image, ground truth and model output
def imshow_mult(imgs, titles=None):
    fig = plt.figure(figsize=(7, 5))
    rows = 1
    cols = len(imgs)
    for i in range(cols):
        fig.add_subplot(rows, cols, i+1)
        imshow(imgs[i])
        plt.axis('off')
        title = 'figure'+str(i+1) if titles is None else titles[i]
        plt.title(title)
    plt.show()


# Input image transforms
imgTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

# Mask transforms
maskTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            ])

def listToString(arr):
    s = ''.join([str(item) for item in arr])
    return s

# rgb labels to int: BW-000, HD-001, PF-010, WR-011, RO-100, RI-101, FV-110, SR-111
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
    def __init__(self, images, masks):
        self.images = [imgTransform(img) for img in images]
        self.masks = [maskTransform(label_mask(mask)) for mask in masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]
