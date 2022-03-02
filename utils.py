import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]

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

# Show example: input image, mask and output
def show_seg(model, loader):
    model.eval()
    img_batch, mask_batch = next(iter(loader))
    viz = visualize_seg(img_batch[0], model(img_batch)[0], mask_batch[0])
    plt.imshow(viz)
    plt.show()