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
        self.features = [4, 8, 16]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # maxpool
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        in_channels = 3
        for feature in self.features:
            self.encoder.append(nn.Conv2d(in_channels, feature, 3, padding=1))
            in_channels = feature

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x