"""Image augmentation and a PyTorch Dataset for the CSVs written by preprocess.py.

`augment()` is what preprocess.py uses to write augmented copies of the train
images to disk. `train_transforms()` can additionally augment on the fly during
training (a different random version every epoch).

    from augmentation import XrayDataset, train_transforms, eval_transforms
    train_ds = XrayDataset("data/processed/train.csv", num_classes, train_transforms())
    val_ds   = XrayDataset("data/processed/val.csv",   num_classes, eval_transforms())

Preview what the augmentation does to one image: set the path at the bottom and run
    python augmentation.py
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

# ImageNet stats; swap for the model's own processor values if they differ.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def augment(hflip=True):
    """Mild, anatomy-preserving augmentation for radiographs. PIL in, PIL out, same size.

    hflip: mirrors left/right. Fine for limbs and lateral views; turn it off for
    tasks where side matters (e.g. heart position on VD/DV thorax views).
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5 if hflip else 0.0),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),  # exposure differences between machines
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
    ])


def train_transforms(size=224, hflip=True):
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        augment(hflip),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def eval_transforms(size=224):
    return T.Compose([
        T.ToImage(),
        T.Resize((size, size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


class XrayDataset(Dataset):
    """Returns (image tensor [3, H, W], multi-hot target [num_classes])."""

    def __init__(self, csv_path, num_classes, transform=None, image_col="image_path"):
        df = pd.read_csv(csv_path, dtype={"label_ids": str})
        self.paths = df[image_col].tolist()
        self.targets = torch.zeros(len(df), num_classes)
        for row, ids in enumerate(df["label_ids"]):
            self.targets[row, [int(i) for i in ids.split("|")]] = 1.0
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # grayscale -> 3 channels for pretrained models
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


if __name__ == "__main__":
    # preview: save 8 augmented versions of one image side by side
    img = Image.open("data/raw/example.png").convert("RGB")
    tf = train_transforms(224)
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    grid = Image.new("RGB", (224 * 8, 224))
    for k in range(8):
        x = (tf(img) * std + mean).clamp(0, 1)
        grid.paste(T.functional.to_pil_image(x), (k * 224, 0))
    grid.save("augmentation_preview.png")
