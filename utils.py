import torch
import matplotlib.pyplot as plt
import torchvision
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
def show_seg(model, loader):
    model.eval()
    img_batch, mask_batch = next(iter(loader))
    output = torch.argmax(model(img_batch), 1)
    imshow_mult([img_batch[0], mask_batch[0], output[0]], ['Input', 'Ground truth', 'Prediction'])

def visualize_seg(image, seg_pred, seg_true):
    def t(img):
        return (img * 127 + 127).to(torch.uint8)

    colors = ['#000000', '#0000FF', '#00FF00', '#00FFFF', '#FF0000', '#FF00FF', '#FFFF00', '#FFFFFF']

    image = t(image)

    seg_true = torchvision.utils.draw_segmentation_masks(
        image,
        torch.nn.functional.one_hot(seg_true, num_classes=8).to(torch.bool).movedim(-1, 0),
        alpha=1.0,
        colors=colors
    )
    seg_pred = torchvision.utils.draw_segmentation_masks(
        image,
        torch.nn.functional.one_hot(seg_pred.argmax(0), num_classes=8).to(torch.bool).movedim(-1, 0),
        alpha=1.0,
        colors=colors
    )
    g = torchvision.utils.make_grid([image, seg_true, seg_pred])
    return g.moveaxis(0, -1)
