from comet_ml import Experiment
import os
import torch
import torch.nn as nn
from torchmetrics import IoU, Accuracy
from torch.utils.data import DataLoader, random_split
from utils import imgDataset, imshow_mult
from models import ConvNet, UNet
from preprocess import preprocess_images
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

def prepare_data(preprocess=False):
    if preprocess:
        preprocess_images(os.environ['PATH'])
    prep_data = torch.load('preprocessed_128.pt')
    trainval_imgs, trainval_masks = prep_data['images'], prep_data['masks']

    # Make dataset and apply transforms
    trainval_data = imgDataset(trainval_imgs, trainval_masks)
    train_size = int(0.8 * len(trainval_data))
    val_size = len(trainval_data) - train_size
    train_data, val_data = random_split(trainval_data, [train_size, val_size])
    #test_data = imgDataset(test_imgs, test_masks)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=True)
    #test_loader = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False)

    return train_loader, val_loader

def train_step(data, target, optimizer, criterion, train_on_gpu=False):
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    target = torch.squeeze(target, dim=1)
    optimizer.zero_grad()

    output = net(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(data, target, criterion, metrics, train_on_gpu=False):
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    target = torch.squeeze(target, dim=1)

    output = net(data)
    loss = criterion(output, target)

    for metric in metrics:
        metric(output, target)

    return loss.item()

# Select model: 'unet', 'conv'
model = 'conv'
# Check if cuda is available
train_on_gpu = torch.cuda.is_available()
print('Is cuda available?', 'Yes' if train_on_gpu else 'No')

train_loader, val_loader = prepare_data()

if model == 'unet':
    net = UNet()
else:
    net = ConvNet()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=params["learning_rate"])

# Metrics
iou = IoU(num_classes=8)
accuracy = Accuracy(num_classes=8)
metrics = [iou, accuracy]

# If cuda is available
if train_on_gpu:
    print('Training on GPU ...')
    model.cuda()
    criterion.cuda()
    iou.cuda()
    accuracy.cuda()

train_loss, val_loss = [], []
epochs = params["epochs"]
for epoch in range(epochs):
    running_loss, val_running_loss = 0., 0.

    net.train()
    for image, mask in train_loader:
        running_loss += train_step(image, mask, optimizer, criterion)

    net.eval()
    for image, mask in val_loader:
        val_running_loss += validation_step(image, mask, criterion, metrics)

    if epoch%5 == 0:
        output = net(image)
        top_class = torch.argmax(output, 1)
        experiment.log_image(top_class[0], name='output')
    experiment.log_metric('IoU', iou.compute())
    experiment.log_metric('val_accuracy', accuracy.compute())

    train_loss.append(running_loss / len(train_loader.dataset))
    val_loss.append(val_running_loss/len(val_loader.dataset))

    print('[epoch', epoch+1, '] Training loss: %.5f' %train_loss[-1], ' Validation loss: %.5f' %val_loss[-1])
    print('         Accuracy: %.2f' %accuracy.compute())

# Show example: input image, mask and output
net.eval()
img_batch, mask_batch = next(iter(val_loader))
output = torch.argmax(net(img_batch), 1)
img, mask, pred = img_batch[1], mask_batch[1], output[1]
imshow_mult([img, mask, pred], ['Input', 'Label', 'Prediction'])

# Plots
epochs_arr = [i+1 for i in range(epochs)]
plt.plot(epochs_arr, train_loss, label='Training loss')
plt.plot(epochs_arr, val_loss, label='Validation loss')
plt.show()
