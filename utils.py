import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def imshow(img):
    img = torch.squeeze(img)

    imshape = list(img.shape)
    if imshape[0] == 3:         # RGB image
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img.numpy())
    elif len(imshape) == 2:     # grayscale image
        img = torch.squeeze(img)
        n_classes = 7
        plt.imshow(img.numpy(), cmap='gray', vmin=0, vmax=n_classes)

# Show input image, ground truth and model output
def imshow_mult(imgs, titles=None):
    fig = plt.figure(figsize=(8, 3))
    rows = 1
    cols = len(imgs)
    for i in range(cols):
        fig.add_subplot(rows, cols, i+1)
        imshow(imgs[i])
        plt.axis('off')
        title = 'figure'+str(i+1) if titles is None else titles[i]
        plt.title(title)
    plt.show()

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]

# Show example: input image, mask and output
def show_example(model, loader):
    model.eval()
    img_batch, mask_batch = next(iter(loader))
    output = torch.argmax(model(img_batch), 1)
    imshow_mult([img_batch[0], mask_batch[0], output[0]], ['Input', 'Label', 'Prediction'])
