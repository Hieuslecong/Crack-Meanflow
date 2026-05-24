import csv
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


def load_split_csv(path) -> List[Tuple[str, str, str]]:
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [(row["name"], row["image_path"], row["mask_path"]) for row in reader]


def _official_group(name: str):
    lower = name.lower()
    if "_train_" in lower:
        return "train"
    if "_test_" in lower:
        return "test"
    if "_valid_" in lower or "_val_" in lower:
        return "val"
    return None


def split_pairs(pairs, train_ratio=0.8, val_ratio=0.1, seed=42):
    official = {"train": [], "val": [], "test": []}
    unlabeled = []
    for pair in pairs:
        group = _official_group(pair[0])
        if group is None:
            unlabeled.append(pair)
        else:
            official[group].append(pair)

    if official["train"] and (official["test"] or official["val"]):
        train_pairs = sorted(official["train"], key=lambda x: x[0])
        val_pairs = sorted(official["val"], key=lambda x: x[0])
        test_pairs = sorted(official["test"], key=lambda x: x[0])
        if not val_pairs and len(train_pairs) >= 3:
            rng = random.Random(seed)
            shuffled = list(train_pairs)
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
            n_val = min(n_val, len(shuffled) - 1)
            val_pairs = sorted(shuffled[:n_val], key=lambda x: x[0])
            val_names = {name for name, _, _ in val_pairs}
            train_pairs = [pair for pair in train_pairs if pair[0] not in val_names]
        return train_pairs, val_pairs, test_pairs, "official_name_split"

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * float(train_ratio))))
    n_remaining = max(1, n_total - n_train)
    n_val = max(1, int(round(n_total * float(val_ratio)))) if n_total >= 3 else 0
    n_val = min(n_val, n_remaining - 1) if n_remaining > 1 else 0
    n_test_start = n_train + n_val
    train_pairs = sorted(shuffled[:n_train], key=lambda x: x[0])
    val_pairs = sorted(shuffled[n_train:n_test_start], key=lambda x: x[0])
    test_pairs = sorted(shuffled[n_test_start:], key=lambda x: x[0])
    if not test_pairs and val_pairs:
        test_pairs = [val_pairs.pop()]
    return train_pairs, val_pairs, test_pairs, "random_seed_42"


def deterministic_split(pairs, train_ratio=0.8, val_ratio=0.1, seed=42):
    train_pairs, _val_pairs, test_pairs, _policy = split_pairs(pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return train_pairs, test_pairs


def write_split_report(path, pairs, train_pairs, val_pairs, test_pairs, image_dir, mask_dir, policy, seed=42):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leakage = {
        "train_has_test_names": [name for name, _, _ in train_pairs if "_test_" in name.lower()],
        "test_has_train_names": [name for name, _, _ in test_pairs if "_train_" in name.lower()],
        "val_has_test_names": [name for name, _, _ in val_pairs if "_test_" in name.lower()],
        "val_has_train_names": [name for name, _, _ in val_pairs if "_train_" in name.lower()],
    }
    text = [
        "# CrackMeanFlow Data Split Report",
        "",
        f"Image dir: `{image_dir}`",
        f"Mask dir: `{mask_dir}`",
        f"Matched pairs: {len(pairs)}",
        f"Train pairs: {len(train_pairs)}",
        f"Val pairs: {len(val_pairs)}",
        f"Test pairs: {len(test_pairs)}",
        f"Split policy: {policy}",
        f"Seed: {seed}",
        "",
        "## Leakage audit",
        f"- train contains `_test_`: {len(leakage['train_has_test_names'])}",
        f"- test contains `_train_`: {len(leakage['test_has_train_names'])}",
        f"- val contains `_test_`: {len(leakage['val_has_test_names'])}",
        f"- val contains `_train_`: {len(leakage['val_has_train_names'])}",
        "",
        "## First train names",
        *[f"- {name}" for name, _, _ in train_pairs[:10]],
        "",
        "## First val names",
        *[f"- {name}" for name, _, _ in val_pairs[:10]],
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
