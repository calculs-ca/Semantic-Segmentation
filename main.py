from comet_ml import Experiment
import os
import torch
import torch.nn as nn
from torchmetrics import IoU, Accuracy
from torch.utils.data import DataLoader, random_split
from utils import imgDataset, show_seg, visualize_seg
from models import ConvNet, UNet
from preprocess import preprocess_images
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
"""
# Select model: 'unet', 'conv'
model = 'conv'
# Hyperparameters
params = {
    "model": model,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 10
}
# Create comet experiment
experiment = Experiment(
    api_key=os.environ['API_KEY'],
    project_name="semantic-segmentation",
    workspace=os.environ['WORKSPACE'],
    disabled=False
)
experiment.log_parameters(params)

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()
print('Is cuda available?', 'Yes' if train_on_gpu else 'No')

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

def train_step(model, data, target, optimizer, criterion):
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    target = torch.squeeze(target, dim=1)
    optimizer.zero_grad()

    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(model, data, target, criterion, metrics):
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    target = torch.squeeze(target, dim=1)

    output = model(data)
    loss = criterion(output, target)

    for metric in metrics:
        metric(output, target)

    return loss.item()

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()

    running_loss = 0.
    for image, mask in train_loader:
        running_loss += train_step(model, image, mask, optimizer, criterion)
    return running_loss

def validation_epoch(model, val_loader, criterion, metrics):
    model.eval()

    val_running_loss = 0.
    for image, mask in val_loader:
        val_running_loss += validation_step(model, image, mask, criterion, metrics)
    return val_running_loss

def train(model, train_loader, val_loader):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Metrics
    iou = IoU(num_classes=8)
    accuracy = Accuracy(num_classes=8)
    metrics = [iou, accuracy]

    # If cuda is available train on gpu
    if train_on_gpu:
        print('Training on GPU ...')
        model.cuda()
        criterion.cuda()
        iou.cuda()
        accuracy.cuda()

    train_loss, val_loss = [], []
    for epoch in range(params["epochs"]):
        running_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_running_loss = validation_epoch(model, val_loader, criterion, metrics)

        if epoch % 10 == 0:
            image, mask = next(iter(val_loader))
            output = model(image)
            mask = torch.squeeze(mask)
            for i in range(min(10, len(image))):
                viz = visualize_seg(image[i], output[i], mask[i])
                experiment.log_image(viz, name='val', step=epoch)

        experiment.log_metric('IoU', iou.compute())
        experiment.log_metric('val_accuracy', accuracy.compute())

        train_loss.append(running_loss / len(train_loader.dataset))
        val_loss.append(val_running_loss / len(val_loader.dataset))

        experiment.log_metric('train_loss', train_loss[-1])
        experiment.log_metric('val_loss', val_loss[-1])

        print('[epoch', epoch + 1, '] Training loss: %.5f' % train_loss[-1], ' Validation loss: %.5f' % val_loss[-1])
        print('         Accuracy: %.2f' % accuracy.compute())

    # Plots
    epochs_arr = [i + 1 for i in range(params["epochs"])]
    plt.plot(epochs_arr, train_loss, label='Training loss')
    plt.plot(epochs_arr, val_loss, label='Validation loss')
    plt.show()

def main():
    # Prepare data
    train_loader, val_loader = prepare_data()
    # Initialize model
    if model == 'unet':
        net = UNet()
    else:
        net = ConvNet()
    # Train model
    train(net, train_loader, val_loader)
    # Show prediction example: input, mask, prediction
    show_seg(net, val_loader)

if __name__ == '__main__':
    main()