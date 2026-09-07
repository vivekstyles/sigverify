"""
dataset.py — Contrastive pair dataset for Siamese signature verification.

Generates pairs of signature images with labels:
  - label=1 (positive): two genuine signatures from the SAME user
  - label=0 (negative): genuine + forgery from the same user, or
                         genuine signatures from DIFFERENT users
"""

import os
import random
from itertools import combinations

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_user_images(dataset_root, user_ids=None):
    """
    Load image paths organized by user and type.

    Returns:
        dict: {user_id: {"genuine": [paths...], "forgery": [paths...]}}
    """
    data = {}
    all_users = sorted(os.listdir(dataset_root))

    if user_ids is not None:
        all_users = [u for u in all_users if u in user_ids]

    for user in all_users:
        user_dir = os.path.join(dataset_root, user)
        if not os.path.isdir(user_dir):
            continue

        genuine_dir = os.path.join(user_dir, "genuine", "Images")
        forgery_dir = os.path.join(user_dir, "skilled forgery", "Images")

        genuine_paths = []
        forgery_paths = []

        if os.path.isdir(genuine_dir):
            genuine_paths = sorted([
                os.path.join(genuine_dir, f)
                for f in os.listdir(genuine_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

        if os.path.isdir(forgery_dir):
            forgery_paths = sorted([
                os.path.join(forgery_dir, f)
                for f in os.listdir(forgery_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

        data[user] = {"genuine": genuine_paths, "forgery": forgery_paths}

    return data


class SignaturePairDataset(Dataset):
    """
    Generates balanced contrastive pairs for Siamese network training.

    Pair types:
      - Positive (label=1): genuine_i + genuine_j from the same user
      - Negative-forgery (label=0): genuine + forgery from the same user
      - Negative-cross-user (label=0): genuine_A + genuine_B from different users
    """

    IMG_HEIGHT = 155
    IMG_WIDTH = 220

    def __init__(self, dataset_root, user_ids=None, pairs_per_epoch=1000,
                 augment=False, seed=42):
        """
        Args:
            dataset_root: Path to Dataset/Dataset/ directory.
            user_ids: List of user IDs to include (None = all).
            pairs_per_epoch: Number of pairs to generate per epoch.
            augment: Whether to apply data augmentation.
            seed: Random seed for reproducibility.
        """
        self.data = load_user_images(dataset_root, user_ids)
        self.user_ids = list(self.data.keys())
        self.pairs_per_epoch = pairs_per_epoch
        self.augment = augment
        self.rng = random.Random(seed)

        # Base transform: grayscale, resize, tensor
        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH)),
            transforms.ToTensor(),  # [0, 1] range
        ])

        # Augmented transform for training
        self.aug_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.IMG_HEIGHT + 20, self.IMG_WIDTH + 20)),
            transforms.RandomCrop((self.IMG_HEIGHT, self.IMG_WIDTH)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=3,
            ),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)),
        ])

        # Generate initial pairs
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        """Generate a balanced set of contrastive pairs."""
        pairs = []

        # Target: ~1/3 positive, ~1/3 forgery-negative, ~1/3 cross-user-negative
        n_each = self.pairs_per_epoch // 3

        # --- Positive pairs: same-user genuine-genuine ---
        for _ in range(n_each):
            user = self.rng.choice(self.user_ids)
            genuine = self.data[user]["genuine"]
            if len(genuine) < 2:
                continue
            img1, img2 = self.rng.sample(genuine, 2)
            pairs.append((img1, img2, 1.0))

        # --- Negative pairs: same-user genuine-forgery ---
        for _ in range(n_each):
            user = self.rng.choice(self.user_ids)
            genuine = self.data[user]["genuine"]
            forgery = self.data[user]["forgery"]
            if len(genuine) == 0 or len(forgery) == 0:
                continue
            img1 = self.rng.choice(genuine)
            img2 = self.rng.choice(forgery)
            pairs.append((img1, img2, 0.0))

        # --- Negative pairs: cross-user genuine-genuine ---
        remaining = self.pairs_per_epoch - len(pairs)
        for _ in range(remaining):
            if len(self.user_ids) < 2:
                break
            user_a, user_b = self.rng.sample(self.user_ids, 2)
            genuine_a = self.data[user_a]["genuine"]
            genuine_b = self.data[user_b]["genuine"]
            if len(genuine_a) == 0 or len(genuine_b) == 0:
                continue
            img1 = self.rng.choice(genuine_a)
            img2 = self.rng.choice(genuine_b)
            pairs.append((img1, img2, 0.0))

        self.rng.shuffle(pairs)
        return pairs

    def reshuffle(self):
        """Re-generate pairs for a new epoch."""
        self.pairs = self._generate_pairs()

    def _load_image(self, path):
        """Load and transform a single image."""
        img = Image.open(path).convert("RGB")
        if self.augment:
            return self.aug_transform(img)
        return self.base_transform(img)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)
        label = torch.tensor(label, dtype=torch.float32)
        return img1, img2, label
