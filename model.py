"""
model.py — Siamese encoder for signature verification.

Architecture:
  - Backbone: ResNet-18 (pretrained), modified for 1-channel grayscale input
  - Embedding head: GAP → FC(512→256) → ReLU → FC(256→128) → L2-normalize
  - Contrastive loss for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SignatureEncoder(nn.Module):
    """
    Encodes a signature image into a 128-dimensional normalized embedding.

    Uses a ResNet-18 backbone pretrained on ImageNet, with the first conv layer
    adapted for single-channel (grayscale) input and the classification head
    replaced with a compact embedding projection.
    """

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt first conv layer: 3-channel → 1-channel
        # Average the pretrained weights across the input channel dimension
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Copy remaining backbone layers (excluding fc)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool  # Global Average Pooling

        # Embedding projection head
        self.embedding_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 1, 155, 220) — grayscale signature images.

        Returns:
            Tensor of shape (B, 128) — L2-normalized embeddings.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 512)

        x = self.embedding_head(x)  # (B, 128)
        x = F.normalize(x, p=2, dim=1)  # L2-normalize

        return x


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.

    L = y * d² + (1 - y) * max(0, margin - d)²

    Where:
      - y = 1 for positive pairs (same signer), 0 for negative pairs
      - d = Euclidean distance between embeddings
      - margin = minimum distance for negative pairs
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: (B, D) embeddings from branch 1
            embedding2: (B, D) embeddings from branch 2
            label: (B,) — 1.0 for positive pairs, 0.0 for negative pairs

        Returns:
            Scalar loss value.
        """
        distance = F.pairwise_distance(embedding1, embedding2, p=2)

        # Positive pairs: minimize distance
        positive_loss = label * distance.pow(2)

        # Negative pairs: push apart beyond margin
        negative_loss = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = (positive_loss + negative_loss).mean()
        return loss


class SiameseNetwork(nn.Module):
    """
    Siamese wrapper that passes two images through a shared encoder
    and computes contrastive loss.
    """

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        self.encoder = SignatureEncoder(
            embedding_dim=embedding_dim, pretrained=pretrained
        )

    def forward(self, img1, img2):
        """
        Args:
            img1, img2: (B, 1, H, W) signature images.

        Returns:
            emb1, emb2: (B, D) embedding vectors.
        """
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        return emb1, emb2
