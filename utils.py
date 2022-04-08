import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as func_transforms
from torch.utils.data import Dataset

IMG_SIZE = 128

# Images dataset
class imgDataset(Dataset):
    def __init__(self, images, masks, use_da=False):
        self.images = images
        self.masks = masks
        self.use_da = use_da

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        gt = self.masks[i]

        if self.use_da:
            # Apply transforms (data augmentation)
            params = transforms.RandomAffine.get_params(
                [-45, 45],  # Rotation: -30, 30 degrees
                None,  # Translation
                [0.8, 1.2],  # Scale
                None,  # Shear
                img_size=[IMG_SIZE, IMG_SIZE]
            )
            img = func_transforms.affine(img, *params, interpolation=transforms.InterpolationMode.BILINEAR)
            gt = func_transforms.affine(gt.unsqueeze(0), *params, interpolation=transforms.InterpolationMode.NEAREST)
            gt = gt[0]

        return img, gt

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