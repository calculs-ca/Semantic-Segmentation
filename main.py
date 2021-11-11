from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from utils import load_images, imgDataset, imshow_mult, visualize_seg
from models import ConvNet, UNet
import torchmetrics
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
Using 50 for training and 25 for testing
"""

# Select model: 'unet', 'conv'
model = 'unet'

experiment = Experiment(project_name="Karen-Semantic-Seg", disabled=False)
params = {
    'epochs': 100,
    'lr': 0.01,
    'wd': 0,
    'model': model,
    'features': [32, 64, 128],
    'limit_train_samples': 0  # 0 will use full train dataset, 4 will use 4 samples.
}
experiment.log_parameters(params)

# Load images from folder
train_imgs = load_images('data/train/images')
train_masks = load_images('data/train/masks')
test_imgs = load_images('data/test/images')
test_masks = load_images('data/test/masks')

# Make dataset and apply transforms
train_data = imgDataset(train_imgs, train_masks)
test_data = imgDataset(test_imgs, test_masks)
if params['limit_train_samples']:
    print('WARNING: Limiting train samples to:', params['limit_train_samples'])
    train_data = Subset(train_data, range(params['limit_train_samples']))

# Data loaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

if model == 'unet':
    net = UNet(params['features'])
else:
    net = ConvNet()
#print(net)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['wd'])

# Metrics
accu_train = torchmetrics.Accuracy(num_classes=8).cuda()
iou_train = torchmetrics.IoU(num_classes=8, absent_score=1.0).cuda()
accu_test = torchmetrics.Accuracy(num_classes=8).cuda()
iou_test = torchmetrics.IoU(num_classes=8, absent_score=1.0).cuda()

net.cuda()
criterion.cuda()

train_loss, test_loss = [], []
accuracy = []
for epoch in range(params['epochs']):
    net.train()
    running_loss = 0.

    accu_train.reset()
    iou_train.reset()

    for image, mask in train_loader:
        image = image.cuda()
        mask = torch.squeeze(mask, dim=1).cuda()
        optimizer.zero_grad()

        output = net(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        accu_train(output, mask)
        iou_train(output, mask)

    experiment.log_metric('accu_train', accu_train.compute(), step=epoch)
    experiment.log_metric('iou_train', iou_train.compute(), step=epoch)

    viz = visualize_seg(image[0].cpu(), output[0].cpu(), mask[0].cpu())
    experiment.log_image(viz, step=epoch)

    train_loss.append(running_loss/len(train_loader.dataset))

    net.eval()
    correct_class = 0
    test_running_loss = 0

    accu_test.reset()
    iou_test.reset()

    for image, mask in test_loader:
        image = image.cuda()
        mask = torch.squeeze(mask, dim=1).cuda()
        output = net(image)
        loss = criterion(output, mask)
        test_running_loss += loss.item()

        accu_test(output, mask)
        iou_test(output, mask)

        total_pixels = image.size()[-1] * image.size()[-2]
        top_class = torch.argmax(output, 1)
        batch_size = image.size()[0]

        for i in range(batch_size):
            m = mask[i]
            equals = top_class[i] == m.view(*top_class[i].shape)

            #print('Correct class:', equals.sum().item(), '/', total_pixels)
            val = (equals.sum().item()*100)/total_pixels
            correct_class += val
            #print('Accuracy: %.2f' %val)
    test_loss.append(test_running_loss/len(test_loader.dataset))
    accuracy.append(correct_class/len(test_loader.dataset))

    experiment.log_metric('accu_test', accu_test.compute(), step=epoch)
    experiment.log_metric('iou_test', iou_test.compute(), step=epoch)

    print('[epoch', epoch+1, '] Training loss: %.5f' %train_loss[-1], ' Validation loss: %.5f' %test_loss[-1])
    print('     Accuracy: %.2f' %accuracy[-1], '%')
