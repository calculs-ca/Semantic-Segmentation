import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted
from tqdm import tqdm

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

# 1 0 1
# 4 2 1
# 4 0 1 = 5

def label_mask(mask):
    return (mask.astype(bool) * [4, 2, 1]).sum(axis=-1)

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks):
        self.images = [imgTransform(img) for img in tqdm(images)]
        self.masks = [maskTransform(label_mask(mask)) for mask in tqdm(masks)]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]


def visualize_seg(image, seg_pred, seg_true):
    def t(img):
        return (img * 127 + 127).to(torch.uint8)

    colors = ['#000000', '#0000FF', '#00FF00', '#00FFFF', '#FF0000', '#FF00FF', '#FFFF00', '#FFFFFF']

    image = t(image)

    seg_true = torchvision.utils.draw_segmentation_masks(
        image,
        torch.nn.functional.one_hot(seg_true, num_classes=8).to(torch.bool).transpose(-1, 0), colors=colors
    )
    seg_pred = torchvision.utils.draw_segmentation_masks(
        image,
        torch.nn.functional.one_hot(seg_pred.argmax(0), num_classes=8).to(torch.bool).transpose(-1, 0), colors=colors
    )
    g = torchvision.utils.make_grid([seg_true, seg_pred])
    return g.moveaxis(0, -1)


