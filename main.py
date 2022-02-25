from comet_ml import Experiment
import os
import torch
import torch.nn as nn
from torchmetrics import IoU, Accuracy
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
import optuna
from utils import imgDataset, show_seg, visualize_seg
from models import ConvNet, UNet
from preprocess import preprocess_images
"""
Dataset: Underwater imagery (SUIM)
"""
# Select model: 'unet', 'convnet'
model_arch = 'convnet'
# Default hyperparameters
def_params = {
    "model": model_arch,
    "features": [64, 128, 256],
    "batch_norm": True,
    "learning_rate": 1.10e-4,
    "batch_size": 64,
    "epochs": 3
}

# Lightning module
class LitModel(pl.LightningModule):
    def __init__(self, model_arch, params, experiment):
        super().__init__()
        if model_arch == 'unet':
            self.model = UNet(params["features"])
        else:
            self.model = ConvNet(params["features"])
        self.experiment = experiment
        self.lr = params['learning_rate']
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.accu_train = Accuracy(num_classes=8)
        self.accu_val = Accuracy(num_classes=8)
        self.iou_train = IoU(num_classes=8)
        self.iou_val = IoU(num_classes=8)

    def training_step(self, batch, batch_idx):
        data, target = batch
        target = torch.squeeze(target, dim=1)

        output = self.model(data)
        loss = self.criterion(output, target)
        self.iou_train(output, target)
        self.accu_train(output, target)
        # Elements for visualization
        viz = [data, output, target]

        return {"loss": loss, "viz": viz}

    def training_epoch_end(self, training_step_outputs):
        # Compute average
        loss = np.mean([out['loss'].item() for out in training_step_outputs])
        # Log metrics to Comet
        self.experiment.log_metric('train_loss', loss, step=self.current_epoch)
        self.experiment.log_metric('train_IoU', self.iou_train.compute(), step=self.current_epoch)
        self.experiment.log_metric('train_accuracy', self.accu_train.compute(), step=self.current_epoch)
        # Reset metrics
        self.iou_train.reset()
        self.accu_train.reset()
        # Log segmentation visualization
        if self.current_epoch%10 == 0:
            visualize = training_step_outputs[-1]["viz"]
            data, output, target = visualize
            target_ = torch.squeeze(torch.clone(target), dim=1)
            for i in range(min(10, len(data))):
                viz = visualize_seg(data[i], output[i], target_[i])
                self.experiment.log_image(viz, name='seg_vis', step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        data, target = batch
        target = torch.squeeze(target, dim=1)

        output = self.model(data)
        loss = self.criterion(output, target)
        self.iou_val(output, target)
        self.accu_val(output, target)
        # Elements for visualization
        viz = [data, output, target]

        return {"loss": loss, "viz": viz}

    def validation_epoch_end(self, validation_step_outputs):
        # Compute average
        loss = np.mean([out['loss'].item() for out in validation_step_outputs])
        # Log metrics average to Comet
        self.experiment.log_metric('val_loss', loss, step=self.current_epoch)
        self.experiment.log_metric('val_IoU', self.iou_val.compute(), step=self.current_epoch)
        self.experiment.log_metric('val_accuracy', self.accu_val.compute(), step=self.current_epoch)
        self.log('val_iou', self.iou_val.compute())
        # Reset metrics
        self.iou_val.reset()
        self.accu_val.reset()
        # Log segmentation visualization
        if self.current_epoch%10 == 0:
            visualize = validation_step_outputs[-1]["viz"]
            data, output, target = visualize
            target_ = torch.squeeze(torch.clone(target), dim=1)
            for i in range(min(10, len(data))):
                viz = visualize_seg(data[i], output[i], target_[i])
                self.experiment.log_image(viz, name='val_seg_vis', step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()
print('Is cuda available?', 'Yes' if train_on_gpu else 'No')

def prepare_data(preprocess=False):
    if preprocess:
        preprocess_images(os.environ['PATH'])
    prep_data = torch.load('./preprocessed_128.pt')
    trainval_imgs, trainval_masks = prep_data['images'], prep_data['masks']
    # Make dataset and apply transforms
    trainval_data = imgDataset(trainval_imgs, trainval_masks)
    train_size = int(0.8 * len(trainval_data))
    val_size = len(trainval_data) - train_size
    train_data, val_data = random_split(trainval_data, [train_size, val_size])

    return train_data, val_data

def train(params, train_data, val_data):
    # Create comet experiment
    experiment = Experiment(
        api_key=os.environ['API_KEY'],
        project_name="semantic-segmentation",
        workspace="aklopezcarbajal",
        disabled=True
    )
    # Log parameters
    experiment.log_parameters(params)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=True)
    # Initialize model
    litmodel = LitModel(model_arch, params, experiment)

    # Train model
    early_stopping = pl.callbacks.EarlyStopping('val_iou')
    trainer = pl.Trainer(logger=True, checkpoint_callback=False, max_epochs=params['epochs'], callbacks=[early_stopping])
    #trainer = pl.Trainer(fast_dev_run=True)    # fast run

    trainer.fit(litmodel, train_loader, val_loader)
    val = trainer.callback_metrics['val_iou']

    experiment.end()
    return val

def objective(trial: optuna.trial.Trial, train_data, val_data):
    params = def_params.copy()
    # Suggest hyperparams
    width = trial.suggest_categorical('width', [32, 64, 96])
    features = [width, 2*width, 4*width]
    params.update({
            'optuna_study': trial.study.study_name,
            'optuna_trial': trial.number,
            'features': features,
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1000),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    })
    trial_val = train(params, train_data, val_data)
    return trial_val

def main():
    train_data, val_data = prepare_data()   # Prepare data

    hp_optim = True
    if hp_optim:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_data, val_data), n_trials=3, timeout=600)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("     ", key, ":", value)
    else:
        train(def_params, train_data, val_data)

if __name__ == '__main__':
    main()