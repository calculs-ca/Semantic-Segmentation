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
accu_train = torchmetrics.Accuracy(num_classes=8)
iou_train = torchmetrics.IoU(num_classes=8, absent_score=1.0)
accu_test = torchmetrics.Accuracy(num_classes=8)
iou_test = torchmetrics.IoU(num_classes=8, absent_score=1.0)

train_loss, test_loss = [], []
accuracy = []
epochs = 30
for epoch in range(epochs):
    net.train()
    running_loss = 0.

    for image, mask in train_loader:
        mask = torch.squeeze(mask, dim=1)
        optimizer.zero_grad()

        output = net(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        accu_train_val = accu_train(output, mask)
        experiment.log_metric('accu_train', accu_train_val)
        iou_train_val = iou_train(output, mask)
        experiment.log_metric('iou_train', iou_train_val)

    train_loss.append(running_loss/len(train_loader.dataset))

    net.eval()
    correct_class = 0
    test_running_loss = 0
    for image, mask in test_loader:
        mask = torch.squeeze(mask, dim=1)
        output = net(image)
        loss = criterion(output, mask)
        test_running_loss += loss.item()

        accu_test_val = accu_test(output, mask)
        experiment.log_metric('accu_test', accu_test_val)
        iou_test_val = iou_test(output, mask)
        experiment.log_metric('iou_test', iou_test_val)

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

    print('[epoch', epoch+1, '] Training loss: %.5f' %train_loss[-1], ' Validation loss: %.5f' %test_loss[-1])
    print('     Accuracy: %.2f' %accuracy[-1], '%')

# Show example: input image, mask and output
img_batch, mask_batch = next(iter(test_loader))
output = torch.argmax(net(img_batch), 1)
img, mask, pred = img_batch[0], mask_batch[0], output[0]
imshow_mult([img, mask, pred], ['Input', 'Label', 'Prediction'])

# Plots
epochs_arr = [i+1 for i in range(epochs)]
plt.figure()

plt.subplot(211)
plt.plot(epochs_arr, train_loss, label='Training loss')
plt.plot(epochs_arr, test_loss, label='Validation loss')
plt.legend()

plt.subplot(212)
plt.plot(epochs_arr, accuracy, label='Accuracy %')
plt.legend()

plt.show()
