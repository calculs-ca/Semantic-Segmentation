import os
from pathlib import Path

from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from utils import load_images, imgDataset, imshow_mult, visualize_seg
from models import ConvNet, UNet
import torchmetrics
import matplotlib.pyplot as plt
import optuna
"""
Dataset: Underwater imagery (SUIM)
Using 50 for training and 25 for testing
"""

# Select model: 'unet', 'conv'
model = 'unet'

params = {
    'do_hpo': False,
    'n_hpo_trials': 100,
    'epochs': 200,
    'batch_size': 16,
    'lr': 1.10e-4,
    'wd': 5.87e-5,
    'model': model,
    'features': [64, 128, 256],
    'limit_train_samples': 0,  # 0 will use full train dataset, 4 will use 4 samples.
    'use_da': True
}


def prepare_data():
    # Load images from folder
    dataset_trainval_path = Path(os.environ['SUIM']) / 'train_val'
    train_imgs = load_images(dataset_trainval_path / 'images')
    train_masks = load_images(dataset_trainval_path / 'masks')

    # Make dataset and apply transforms
    trainval_dataset = imgDataset(train_imgs, train_masks, use_da=params['use_da'])
    val_size = int(len(trainval_dataset) * 0.2)
    train_size = len(trainval_dataset) - val_size
    train_data, test_data = random_split(trainval_dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))
    # TODO disable DA for val/test ?
    if params['limit_train_samples']:
        print('WARNING: Limiting train samples to:', params['limit_train_samples'])
        train_data = Subset(train_data, range(params['limit_train_samples']))

    return train_data, test_data


def train_trial(trial, train_data, test_data):
    """Train the model using the param suggestions from Optuna"""
    width = trial.suggest_categorical('width', [32, 64, 96])
    features = [width, width * 2, width * 4]
    params.update({
        'optuna_study': trial.study.study_name,
        'optuna_trial': trial.number,
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'batch_size': trial.suggest_categorical('batch_size', [16, 24, 32]),
        'wd': trial.suggest_loguniform('wd', 1e-7, 1e-3),
        'features': features
    })
    iou_test_val = train(params, train_data, test_data)
    return iou_test_val


def train(params, train_data, test_data):
    experiment = Experiment(project_name="Karen-Semantic-Seg", disabled=False)
    experiment.log_parameters(params)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False, pin_memory=True)

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

        if epoch % 10 == 0:
            for i in range(4):
                viz = visualize_seg(image[i].cpu(), output[i].cpu(), mask[i].cpu())
                experiment.log_image(viz, name='trn', step=epoch)

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

        iou_test_val = iou_test.compute()
        experiment.log_metric('accu_test', accu_test.compute(), step=epoch)
        experiment.log_metric('iou_test', iou_test_val, step=epoch)

        if epoch % 10 == 0:
            for i in range(10):
                viz = visualize_seg(image[i].cpu(), output[i].cpu(), mask[i].cpu())
                experiment.log_image(viz, name='val', step=epoch)

    experiment.end()
    return iou_test_val


def main():
    print('Preparing data')
    train_data, test_data = prepare_data()

    print('Launching training')
    if params['do_hpo']:
        study = optuna.create_study(storage='sqlite:///optuna.db', direction='maximize')

        def objective(trial):
            return train_trial(trial, train_data, test_data)

        study.optimize(objective, n_trials=params['n_hpo_trials'])  # Maximize Test IoU
        torch.save(study.best_params, 'best_params.pkl')
    else:
        train(params, train_data, test_data)


if __name__ == '__main__':
    main()
