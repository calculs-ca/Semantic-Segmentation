import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_images, imgDataset, imshow_mult
from models import ConvNet, UNet
import matplotlib.pyplot as plt
"""
Dataset: Underwater imagery (SUIM)
Using 50 for training and 25 for testing
"""
# Select model: 'unet', 'conv'
model = 'conv'

# Load images from folder
folder_path = '/home/karen/Documents/data'
train_imgs = load_images(folder_path+'/train/images')
train_masks = load_images(folder_path+'/train/masks')
test_imgs = load_images(folder_path+'/test/images')
test_masks = load_images(folder_path+'/test/masks')

# Make dataset and apply transforms
train_data = imgDataset(train_imgs, train_masks)
test_data = imgDataset(test_imgs, test_masks)

# Data loaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

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
epochs = 30
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
    print('     Accuracy: %.2f' %accuracy[-1], '%')

# Show example: input image, mask and output
img_batch, mask_batch = next(iter(test_loader))
output = torch.argmax(net(img_batch), 1)
img, mask, pred = img_batch[0], mask_batch[0], output[0]
imshow_mult([img, mask, pred], ['Input', 'Label', 'Prediction'])

# Plots
epochs_arr = [i+1 for i in range(epochs)]
plt.figure()

plt.subplot(211)
plt.plot(epochs_arr, train_loss, label='Training loss')
plt.plot(epochs_arr, test_loss, label='Validation loss')
plt.legend()

plt.subplot(212)
plt.plot(epochs_arr, accuracy, label='Accuracy %')
plt.legend()

plt.show()
