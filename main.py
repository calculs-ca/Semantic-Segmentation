from comet_ml import Experiment
import os
import argparse
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
model_arch = 'unet'

# Check if cuda is available
train_on_gpu = torch.cuda.is_available()
print('Is cuda available?', 'Yes' if train_on_gpu else 'No')

# Default hyperparameters
dflt_params = {
    "model": model_arch,
    "features": [64, 128, 256],
    "learning_rate": 1.10e-4,
    "weight_decay": 1.0e-4,
    "batch_size": 32,
    "epochs": 100,
    "gpus": 1 if train_on_gpu else None,
    "use_da": False
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
        self.wd = params['weight_decay']
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.accu_train = Accuracy(num_classes=8)
        self.accu_val = Accuracy(num_classes=8)
        self.iou_train = IoU(num_classes=8)
        self.iou_val = IoU(num_classes=8)

    def training_step(self, batch, batch_idx):
        data, target = batch
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
            for i in range(min(10, len(data))):
                if train_on_gpu:
                    viz = visualize_seg(data[i].cpu(), output[i].cpu(), target[i].cpu())
                else:
                    viz = visualize_seg(data[i], output[i], target[i])
                self.experiment.log_image(viz, name='seg_vis', step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        data, target = batch
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
            for i in range(min(10, len(data))):
                if train_on_gpu:
                    viz = visualize_seg(data[i].cpu(), output[i].cpu(), target[i].cpu())
                else:
                    viz = visualize_seg(data[i], output[i], target[i])
                self.experiment.log_image(viz, name='val_seg_vis', step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

def prepare_data(preprocess=False):
    if preprocess:
        preprocess_images(os.environ['DATA_PATH'])
    prep_data = torch.load(os.environ['DATA_PATH']+'/preprocessed_128.pt')
    trainval_imgs, trainval_masks_ = prep_data['images'], prep_data['masks']
    trainval_masks = [np.squeeze(m) for m in prep_data['masks']]

    # Make dataset and apply transforms
    trainval_data = imgDataset(trainval_imgs, trainval_masks, use_da=dflt_params['use_da'])
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
    val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False)
    # Initialize model
    litmodel = LitModel(model_arch, params, experiment)

    # Train model
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_iou", patience=10, mode="max")
    trainer = pl.Trainer(logger=True, checkpoint_callback=False, max_epochs=params['epochs'], gpus=params['gpus'])
    #trainer = pl.Trainer(fast_dev_run=True)    # Fast run

    trainer.fit(litmodel, train_loader, val_loader)
    val = trainer.callback_metrics['val_iou']

    # Show segmentation example: input, prediction, true segmentation
    #show_seg(litmodel.model, val_loader)
    experiment.end()
    #save model check point
    return val

def objective(trial: optuna.trial.Trial, train_data, val_data):
    params = dflt_params.copy()
    # Suggest hyperparams
    width = trial.suggest_categorical('width', [32, 64, 96])
    features = [width, 2*width, 4*width]
    params.update({
            'optuna_study': trial.study.study_name,
            'optuna_trial': trial.number,
            'features': features,
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 0.1),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    })
    print('update params', params)
    trial_val = train(params, train_data, val_data)
    return trial_val

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hpo', action='store_true', help='Perform an HPO, instead of just doing a single run of training.')
    args = ap.parse_args()
    
    train_data, val_data = prepare_data()   # Prepare data

    hp_optim = False
    if args.hpo:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_data, val_data), n_trials=1)
        torch.save(study.best_params, 'best_params.pkl')
    else:
        train(dflt_params, train_data, val_data)
