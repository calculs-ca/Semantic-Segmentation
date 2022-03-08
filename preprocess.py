import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from natsort import natsorted
"""
Input images have 3 channels RGB
Classes: 8, BW-000, HD-001, PF-010, WR-011, RO-100, RI-101, FV-110, SR-111
"""
def load_images(path):
    images = []
    img_list = natsorted(os.listdir(path))
    for f in img_list:
        img_path = os.path.join(path, f)
        img = np.array(Image.open(img_path).convert('RGB'))
        if img is not None:
            images.append(img)
    return images

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

def label_mask(mask):
    mask = np.sum(mask.astype(bool)*[4, 2, 1], axis=-1)
    return mask

def preprocess_images(path):
    trainval_imgs = load_images(path+'/train_val/images')
    trainval_masks = load_images(path+'/train_val/masks')
    """
    test_imgs = load_images(folder_path+'/TEST/images')
    test_masks = load_images(folder_path+'/TEST/masks')
    """
    images = [imgTransform(img) for img in trainval_imgs]
    masks = [maskTransform(label_mask(mask)) for mask in trainval_masks]
    #Save dictionary
    torch.save({'images': images, 'masks': masks}, 'preprocessed_128.pt')