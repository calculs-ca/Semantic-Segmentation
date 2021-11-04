import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_images, imshow, imgDataset
from models import ConvNet, UNet
"""
Dataset: Underwater imagery (SUIM)
Using 50 for training and 25 for testing
"""
# Select model: 'unet', 'conv'
model = 'conv'

# Load images from folder
train_imgs = load_images('data/train/images')
train_masks = load_images('data/train/masks')
test_imgs = load_images('data/test/images')
test_masks = load_images('data/test/masks')

# Make dataset and apply transforms
train_data = imgDataset(train_imgs, train_masks)
test_data = imgDataset(test_imgs, test_masks)

# Data loaders
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
# Show image sample
img, mask = next(iter(train_loader))
#imshow(mask)

if model == 'unet':
    net = UNet()
else:
    net = ConvNet()
print(net)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

train_loss, test_loss = [], []
accuracy = []
epochs = 2
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
    test_running_loss = 0
    for image, mask in test_loader:
        mask = torch.squeeze(mask, dim=1)
        output = net(image)
        loss = criterion(output, mask)
        test_running_loss += loss.item()

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

    print('[epoch', epoch+1, '] Training loss: %.5f' %train_loss[-1], ' Validation loss: %.5f' %test_loss[-1])
    print('     Accuracy: %.2f' %accuracy[-1])