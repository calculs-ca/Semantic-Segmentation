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
import pytorch_lightning as pl
"""
Dataset: Underwater imagery (SUIM)
"""
# Select model: 'unet', 'conv'
model = 'conv'
# Hyperparameters
params = {
    "model": model,
    "features": [16, 32, 64],
    "batch_norm": True,
    "learning_rate": 0.001,
    "batch_size": 64,
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

# Lightning module
class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.iou = IoU(num_classes=8)
        self.accuracy = Accuracy(num_classes=8)

    def training_step(self, batch, batch_idx):
        data, target = batch
        target = torch.squeeze(target, dim=1)

        output = self.model(data)
        loss = self.criterion(output, target)
        experiment.log_metric('train_loss', loss.item())

        if self.current_epoch%10 == 0 and batch_idx == 0:
            target = torch.squeeze(target)
            for i in range(min(10, len(data))):
                viz = visualize_seg(data[i], output[i], target[i])
                experiment.log_image(viz, name='seg_vis', step=self.current_epoch)

        return loss

    def validation_step(self, batch):
        data, target = batch
        target = torch.squeeze(target, dim=1)

        output = self.model(data)
        loss = self.criterion(output, target)
        experiment.log_metric('val_loss', loss.item())

        self.iou(output, target)
        self.accuracy(output, target)
        experiment.log_metric('IoU', self.iou.compute())
        experiment.log_metric('val_accuracy', self.accuracy.compute())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        return optimizer

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

def main():
    # Prepare data
    train_loader, val_loader = prepare_data()
    # Initialize model
    if model == 'unet':
        net = LitModel(UNet(params["features"]))
    else:
        net = LitModel(ConvNet(params["features"]))
    # Train model
    trainer = pl.Trainer(max_epochs=params["epochs"])
    trainer.fit(net, train_loader)

    # Show prediction example: input, mask, prediction
    show_seg(net.model, val_loader)

if __name__ == '__main__':
    main()