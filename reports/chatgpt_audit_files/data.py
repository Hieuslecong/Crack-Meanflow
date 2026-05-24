from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_pairs(image_dir, mask_dir) -> List[Tuple[str, str, str]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    names = sorted(set(images) & set(masks))
    return [(name, str(images[name]), str(masks[name])) for name in names]


def deterministic_split(pairs, train_ratio=0.8):
    n_train = max(1, int(len(pairs) * float(train_ratio)))
    n_train = min(n_train, len(pairs) - 1) if len(pairs) > 1 else len(pairs)
    return pairs[:n_train], pairs[n_train:]


def write_split_report(path, pairs, train_pairs, test_pairs, image_dir, mask_dir):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# CrackMeanFlow Data Split Report",
        "",
        f"Image dir: `{image_dir}`",
        f"Mask dir: `{mask_dir}`",
        f"Matched pairs: {len(pairs)}",
        f"Train pairs: {len(train_pairs)}",
        f"Test pairs: {len(test_pairs)}",
        "Split policy: deterministic sorted stem split; no random state dependency.",
        "",
        "## First train names",
        *[f"- {name}" for name, _, _ in train_pairs[:10]],
        "",
        "## First test names",
        *[f"- {name}" for name, _, _ in test_pairs[:10]],
    ]
    path.write_text("\n".join(text) + "\n")


class PairedCrackDataset(Dataset):
    def __init__(self, pairs, image_size=256, augment=False):
        self.pairs = list(pairs)
        self.augment = bool(augment)
        self.image_tf = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
        self.mask_tf = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        name, image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
        image = self.image_tf(image)
        mask = (self.mask_tf(mask) > 0.5).float()
        return {"name": name, "crack": image, "mask": mask}
