# data/dataset.py
import os
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

_TINY_MEAN = (0.480, 0.448, 0.398)
_TINY_STD  = (0.277, 0.269, 0.282)

class TinyTrainDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.transform = transform
        self.class_to_idx = {c:i for i,c in enumerate(class_names)}
        self.samples = []
        for cls in class_names:
            img_dir = os.path.join(root_dir, cls, "images")
            if os.path.exists(img_dir):
                for fn in os.listdir(img_dir):
                    if fn.endswith(".JPEG"):
                        self.samples.append((os.path.join(img_dir, fn), self.class_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert("RGB")
        if self.transform: x = self.transform(x)
        return x, y

class TinyValDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.transform = transform
        self.class_to_idx = {c:i for i,c in enumerate(class_names)}
        self.samples = []
        with open(os.path.join(root_dir, "val_annotations.txt")) as f:
            for line in f:
                fn, cls, *_ = line.strip().split('\t')
                self.samples.append((os.path.join(root_dir, "images", fn), self.class_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert("RGB")
        if self.transform: x = self.transform(x)
        return x, y

def _transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_TINY_MEAN, _TINY_STD),
    ])

def _load_classes(root: str):
    with open(os.path.join(root, "wnids.txt")) as f:
        return [line.strip() for line in f]

def create_loaders(root: str, batch_size: int, num_workers: int , img_size: int) -> Tuple[DataLoader, DataLoader, int]:
    classes = _load_classes(root)
    tfm = _transform(img_size)
    train_ds = TinyTrainDataset(os.path.join(root, "train"), classes, tfm)
    val_ds   = TinyValDataset(os.path.join(root, "val"), classes, tfm)
    pin = torch.cuda.is_available()
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=pin)
    val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=pin)
    return train, val, len(classes)