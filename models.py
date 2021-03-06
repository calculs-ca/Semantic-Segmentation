import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Fully Convolutional model
class ConvNet(nn.Module):
    def __init__(self, features = [16, 32, 64]):
        super(ConvNet, self).__init__()
        self.features = features
        out_channels = 8
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # maxpool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # un-pool
        self.unpool = nn.ModuleList()
        # batch norm layers
        self.batch_norm = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)

        in_channels = 3
        for feature in self.features:
            self.encoder.append(nn.Conv2d(in_channels, feature, 3, padding=1))
            in_channels = feature
            self.batch_norm.append(nn.BatchNorm2d(feature))
        self.bottom = nn.Conv2d(self.features[-1], self.features[-1]*2, 3, padding=1)
        # out channels = 8
        for feature in reversed(self.features):
            self.decoder.append(nn.Conv2d(feature*2, feature, 3, padding=1))
            self.unpool.append(nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2))
        self.final = nn.Conv2d(features[0], out_channels, 3, padding=1)

    def forward(self, x):
        for i in range(len(self.features)):
            x = self.encoder[i](x)
            x = self.batch_norm[i](x)
            x = self.relu(x)
            x = self.pool(x)
        x = self.bottom(x)
        for i in range(len(self.features)):
            x = self.decoder[i](x)
            x = self.batch_norm[-1-i](x)
            x = self.relu(x)
            x = self.unpool[i](x)
        x = self.final(x)
        return x

# Double convolution with batch normalization
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

# UNet model
class UNet(nn.Module):
    def __init__(self, features = [16, 32, 64]):
        super(UNet, self).__init__()
        in_channels = 3
        out_channels = 8

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # Bottom layer
        self.bottom = DoubleConv(features[-1], features[-1]*2)
        # Up
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for step in self.downs:
            x = step(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom(x)

        skip_connections.reverse()
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            connection = skip_connections[i // 2]
            if x.shape != connection.shape:
                x = TF.resize(x, size=connection.shape[2:])
            concat_skip = torch.cat((connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        x = self.final_conv(x)
        return x