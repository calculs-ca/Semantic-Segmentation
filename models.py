import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Input images have 3 channels RGB
Classes: 8, BW-000, HD-001, PF-010, WR-011, RO-100, RI-101, FV-110, SR-111
"""
# Fully Convolutional model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = [8, 16, 32]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # maxpool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = 3
        for feature in self.features:
            self.encoder.append(nn.Conv2d(in_channels, feature, 3, padding=1))
            in_channels = feature
        self.bottom = nn.Conv2d(self.features[-1], self.features[-1]*2, 3, padding=1)
        # out channels = 8
        for feature in reversed(self.features):
            self.decoder.append(nn.Conv2d(feature*2, feature, 3, padding=1))
            self.decoder.append(nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2))

    def forward(self, x):
        for step in self.encoder:
            x = step(x)
            x = self.pool(x)
        x = self.bottom(x)
        for step in self.decoder:
            x = step(x)

        return x