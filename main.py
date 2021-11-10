import os
import cv2
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_images, imshow, imgDataset, visualize_seg
from models import ConvNet, UNet
import torchmetrics
"""
Dataset: Underwater imagery (SUIM)
Using 50
"""

experiment = Experiment(project_name="Karen-Semantic-Seg")

# Select model: 'unet', 'conv'
model = 'unet'

# Load images from folder
images = load_images('data/images')
masks = load_images('data/masks')
# Make dataset and apply transforms
img_data = imgDataset(images, masks)
loader = DataLoader(img_data, batch_size=32, shuffle=True)
# Show image sample
img, mask = next(iter(loader))
#imshow(mask)

if model == 'unet':
    net = UNet()
else:
    net = ConvNet()
#print(net)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
accu_train = torchmetrics.Accuracy(num_classes=8)
iou_train = torchmetrics.IoU(num_classes=8)

experiment.log_parameters({
    'model': model
})

epochs = 30
for epoch in range(epochs):
    running_loss = 0.

    for i, (image, mask) in enumerate(loader):
        mask = torch.squeeze(mask, dim=1)
        optimizer.zero_grad()

        output = net(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #experiment.log_metric('loss', )

        accu_train_val = accu_train(output, mask)
        experiment.log_metric('accu_train', accu_train_val)
        iou_train_val = iou_train(output, mask)
        experiment.log_metric('iou_train', iou_train_val)

        if i == 0:
            for im, ou, ma in zip(image, output, mask):
                viz = visualize_seg(im, ou, ma)
                experiment.log_image(viz)

    print('loss:', running_loss)