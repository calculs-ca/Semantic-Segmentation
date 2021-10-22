import os
import cv2
import torch
import matplotlib.pyplot as plt
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
    plt.imshow(img.numpy())
    plt.show()