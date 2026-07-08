"""
Dataset loading, preprocessing, and augmentation for the sugarcane leaf CNN.

Design choices (explained per the master-prompt requirement to justify every
decision):
  - We use torchvision.datasets.ImageFolder as the base loader since the
    sugarcane leaf dataset ships in <class_name>/<image>.jpg structure.
  - We then remap ImageFolder's raw class indices to our 3 condition levels
    via config.RAW_TO_CONDITION, so one Dataset wrapper serves both the
    11-class raw structure and the 3-class problem the RL agent needs.
  - Augmentation (random flip, rotation, color jitter) is applied ONLY to
    the training split. This compensates for the sugarcane-specific dataset
    being much smaller than general-purpose image datasets, without
    artificially inflating validation/test accuracy.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from . import config


class ConditionRemappedDataset(Dataset):
    """Wraps an ImageFolder dataset, remapping raw disease classes to
    3 collapsed condition levels (healthy / moderate_stress / severe_stress).
    """

    def __init__(self, root, transform=None):
        self.base = datasets.ImageFolder(root=root)
        self.transform = transform

        # Build raw_index -> condition_index mapping once.
        raw_classes = self.base.classes  # e.g. ['BandedChlorosis', 'Dried', 'Healthy', ...]
        missing = [c for c in raw_classes if c not in config.RAW_TO_CONDITION]
        if missing:
            raise ValueError(
                f"These raw class folder names are not in RAW_TO_CONDITION in "
                f"config.py -- update the mapping to match your actual dataset: {missing}"
            )

        self.raw_to_condition_idx = {
            i: config.CONDITION_CLASSES.index(config.RAW_TO_CONDITION[cls])
            for i, cls in enumerate(raw_classes)
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, raw_label = self.base[idx]
        condition_label = self.raw_to_condition_idx[raw_label]
        if self.transform:
            image = self.transform(image)
        return image, condition_label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats, since
                             std=[0.229, 0.224, 0.225]),    # we use an ImageNet-pretrained backbone
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_dataloaders(root=None, batch_size=None):
    root = root or config.RAW_DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    train_tf, eval_tf = get_transforms()

    # Load once without transform to get consistent split indices, then
    # attach the right transform to each split via a thin wrapper.
    full_notransform = ConditionRemappedDataset(root, transform=None)
    n = len(full_notransform)
    n_test = int(n * config.TEST_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_subset, val_subset, test_subset = random_split(
        full_notransform, [n_train, n_val, n_test], generator=generator
    )

    class _WithTransform(Dataset):
        """Applies a given transform to a Subset of ConditionRemappedDataset,
        without re-triggering the base (untransformed) dataset's transform."""

        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            base_dataset = self.subset.dataset          # ConditionRemappedDataset (transform=None)
            real_idx = self.subset.indices[idx]
            raw_image, raw_label = base_dataset.base[real_idx]
            condition_label = base_dataset.raw_to_condition_idx[raw_label]
            if self.transform:
                raw_image = self.transform(raw_image)
            return raw_image, condition_label

    train_ds = _WithTransform(train_subset, train_tf)
    val_ds = _WithTransform(val_subset, eval_tf)
    test_ds = _WithTransform(test_subset, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
