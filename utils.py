import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as func_transforms
from torch.utils.data import Dataset
from natsort import natsorted
from tqdm import tqdm

"""
Input images have 3 channels RGB
Classes: 8, BW-000, HD-001, PF-010, WR-011, RO-100, RI-101, FV-110, SR-111
"""

IMG_SIZE = 128

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
    fig = plt.figure(figsize=(6, 5))
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
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

# Mask transforms
maskTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            ])

def label_mask(mask):
    mask = np.sum(mask.astype(bool)*[4, 2, 1], axis=-1)
    return mask

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks, use_da=False):
        self.use_da = use_da
        self.images = [imgTransform(img) for img in tqdm(images)]
        self.masks = [maskTransform(label_mask(mask)) for mask in tqdm(masks)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        gt = self.masks[i]

        if self.use_da:
            # Apply transforms (data augmentation)
            params = transforms.RandomAffine.get_params(
                [-45, 45],  # Rotation: -30, 30 degrees
                None,  # Translation
                [0.8, 1.2],  # Scale
                None,  # Shear
                img_size=[IMG_SIZE, IMG_SIZE]
            )
            img = func_transforms.affine(img, *params, interpolation=transforms.InterpolationMode.BILINEAR)
            gt = func_transforms.affine(gt.unsqueeze(0), *params, interpolation=transforms.InterpolationMode.NEAREST)
            gt = gt[0]

        return img, gt


def visualize_seg(image, seg_pred, seg_true):
    def t(img):
        return (img * 127 + 127).to(torch.uint8)

    colors = ['#000000', '#0000FF', '#00FF00', '#00FFFF', '#FF0000', '#FF00FF', '#FFFF00', '#FFFFFF']

    image = t(image)

    seg_true = torchvision.utils.draw_segmentation_masks(
        image,
        torch.nn.functional.one_hot(seg_true, num_classes=8).to(torch.bool).movedim(-1, 0),
        alpha=1.0,
        colors=colors
    )
    seg_pred = torchvision.utils.draw_segmentation_masks(
        image,
        torch.nn.functional.one_hot(seg_pred.argmax(0), num_classes=8).to(torch.bool).movedim(-1, 0),
        alpha=1.0,
        colors=colors
    )
    g = torchvision.utils.make_grid([image, seg_true, seg_pred])
    return g.moveaxis(0, -1)


