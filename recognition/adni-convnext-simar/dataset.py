import os
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

CLASS_TO_IDX = {"AD": 0, "CN": 1}


@dataclass
class ADNIConfig:
    data_root: str="/home/groups/comp3710/ADNI"
    img_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ADNIDataset(Dataset):
    def __init__(self, data_root: str , split: str, img_size: int = 224):
        self.data_root = data_root
        self.split = split
        self.samples = self.discover_samples()
        self.tform = self._build_transforms(img_size)

    def discover_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for cls in ("AD", "CN"):
            cls_dir = os.path.join(self.data_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        samples.append((os.path.join(root, f), CLASS_TO_IDX[cls]))
        if not samples:
            raise FileNotFoundError(
                f"No images found under {self.data_root}. Expected subfolders AD/ and CN/."
            )
        return samples

    def _build_transforms(self, img_size: int):
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        x = self.tform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, path


def make_splits(cfg: ADNIConfig, seed: int = 42):
    full = ADNIDataset(cfg.data_root, split="train", img_size=cfg.img_size)
    n_total = len(full)
    n_train = int(n_total * cfg.train_ratio)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, lengths=[n_train, n_val], generator=gen)
    # mark transforms appropriately
    train_set.dataset.split = "train"
    val_set.dataset.split = "val"
    return train_set, val_set
