from comet_ml import Experiment
import os
import torch
import torch.nn as nn
from torchmetrics import IoU, Accuracy
from torch.utils.data import DataLoader, random_split
from utils import load_images, imgDataset, imshow_mult
from models import ConvNet, UNet
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
"""
# Create comet experiment
experiment = Experiment(
    api_key=os.environ['API_KEY'],
    project_name="semantic-segmentation",
    workspace=os.environ['WORKSPACE'],
)
# Hyperparameters
params = {
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 10
}
experiment.log_parameters(params)

# Select model: 'unet', 'conv'
model = 'conv'

# Load images from folder
folder_path = os.environ['PATH']
trainval_imgs = load_images(folder_path+'/train/images')
trainval_masks = load_images(folder_path+'/train/masks')
test_imgs = load_images(folder_path+'/test/images')
test_masks = load_images(folder_path+'/test/masks')

# Make dataset and apply transforms
trainval_data = imgDataset(trainval_imgs, trainval_masks)
train_size = int(0.8 * len(trainval_data))
val_size = len(trainval_data) - train_size
train_data, val_data = random_split(trainval_data, [train_size, val_size])
test_data = imgDataset(test_imgs, test_masks)

# Data loaders
train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=True)
test_loader = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False)

if model == 'unet':
    net = UNet()
else:
    net = ConvNet()
print(net)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=params["learning_rate"])

# Metrics
iou = IoU(num_classes=8)
accuracy = Accuracy()
pixel_accuracy = []

train_loss, val_loss = [], []
epochs = params["epochs"]
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
    train_loss.append(running_loss/len(train_loader.dataset))

    net.eval()
    correct_class = 0
    val_running_loss = 0
    for image, mask in val_loader:
        mask = torch.squeeze(mask, dim=1)
        output = net(image)
        loss = criterion(output, mask)
        val_running_loss += loss.item()

        total_pixels = image.size()[-1] * image.size()[-2]
        top_class = torch.argmax(output, 1)
        batch_size = image.size()[0]

        for i in range(batch_size):
            img, m = image[i], mask[i]
            iou_val = iou(top_class[i], m)
            val_acc = accuracy(top_class[i], m)
            experiment.log_metric('IoU', iou_val)
            experiment.log_metric('val_accuracy', val_acc)

            equals = top_class[i] == m.view(*top_class[i].shape)
            pixelacc_val = (equals.sum().item()*100)/total_pixels
            correct_class += pixelacc_val
            #print('Accuracy: %.2f' %pixelacc_val, 'Torch accuracy:', val_acc)
    val_loss.append(val_running_loss/len(val_loader.dataset))
    pixel_accuracy.append(correct_class/len(val_loader.dataset))

    print('[epoch', epoch+1, '] Training loss: %.5f' %train_loss[-1], ' Validation loss: %.5f' %val_loss[-1])
    print('         Accuracy: %.2f' %pixel_accuracy[-1], '%')

# Show example: input image, mask and output
net.eval()
img_batch, mask_batch = next(iter(test_loader))
output = torch.argmax(net(img_batch), 1)
img, mask, pred = img_batch[1], mask_batch[1], output[1]
imshow_mult([img, mask, pred], ['Input', 'Label', 'Prediction'])

# Plots
epochs_arr = [i+1 for i in range(epochs)]
plt.figure()

plt.subplot(211)
plt.plot(epochs_arr, train_loss, label='Training loss')
plt.plot(epochs_arr, val_loss, label='Validation loss')
plt.legend()

plt.subplot(212)
plt.plot(epochs_arr, pixel_accuracy, label='Accuracy %')
plt.legend()

plt.show()
