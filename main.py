from scipy.io import loadmat
import os
import cv2
"""
Dataset: Underwater imagery (SUIM)
Using 50
"""
imgpath = 'data/images'
maskpath = 'data/masks'

images, masks = [], []
for filename in os.listdir(imgpath):
    img = cv2.imread(os.path.join(imgpath, filename))
    if img is not None:
        images.append(img)