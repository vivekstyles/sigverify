"""
verify.py — Signature verification using the trained Siamese encoder.

Usage:
    python verify.py --img1 path/to/sig1.png --img2 path/to/sig2.png
    python verify.py --img1 path/to/sig1.png --img2 path/to/sig2.png --threshold 0.5

Computes embedding distance between two signature images and outputs
a similarity score with an accept/reject decision.
"""

import os
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import SignatureEncoder


# Constants
IMG_HEIGHT = 155
IMG_WIDTH = 220
CHECKPOINT_DIR = "checkpoints"


def load_encoder(checkpoint_path=None, device="cpu"):
    """Load trained encoder from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_encoder.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to train the encoder."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    embedding_dim = checkpoint.get("embedding_dim", 128)

    encoder = SignatureEncoder(embedding_dim=embedding_dim, pretrained=False)
    print(checkpoint["encoder_state_dict"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()

    return encoder


def load_threshold(threshold_path=None):
    """Load the optimal threshold from training."""
    if threshold_path is None:
        threshold_path = os.path.join(CHECKPOINT_DIR, "threshold.pth")

    if not os.path.exists(threshold_path):
        print("Warning: threshold.pth not found, using default threshold 0.5")
        return 0.5

    data = torch.load(threshold_path, map_location="cpu", weights_only=False)
    return float(data["threshold"])


def preprocess_image(image_path):
    """Load and preprocess a signature image for the encoder."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension: (1, 1, H, W)


@torch.no_grad()
def get_embedding(encoder, image_path, device="cpu"):
    """Compute embedding for a single signature image."""
    img_tensor = preprocess_image(image_path).to(device)
    embedding = encoder(img_tensor)
    return embedding


@torch.no_grad()
def verify_signatures(encoder, img1_path, img2_path, threshold=None, device="cpu"):
    """
    Verify if two signatures belong to the same person.

    Args:
        encoder: Trained SignatureEncoder model.
        img1_path: Path to first signature image.
        img2_path: Path to second signature image.
        threshold: Distance threshold for accept/reject. If None, loads from checkpoint.
        device: Device to run inference on.

    Returns:
        dict with keys:
          - distance: Euclidean distance between embeddings
          - similarity: Cosine similarity between embeddings
          - threshold: Threshold used for decision
          - is_match: True if signatures are from the same person
          - confidence: Confidence score (0-1, higher = more confident match)
    """
    if threshold is None:
        threshold = load_threshold()

    emb1 = get_embedding(encoder, img1_path, device)
    emb2 = get_embedding(encoder, img2_path, device)
    print(emb1)
    print(emb2)
    # Euclidean distance
    distance = F.pairwise_distance(emb1, emb2, p=2).item()

    # Cosine similarity
    similarity = F.cosine_similarity(emb1, emb2).item()

    # Decision
    is_match = distance < threshold

    # Confidence: how far the distance is from the threshold
    # Scale so that distance=0 → confidence=1, distance=threshold → confidence=0.5
    confidence = max(0.0, min(1.0, 1.0 - (distance / (2 * threshold))))

    return {
        "distance": distance,
        "similarity": similarity,
        "threshold": threshold,
        "is_match": is_match,
        "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Signatures")
    parser.add_argument("--img1", required=True, help="Path to first signature image")
    parser.add_argument("--img2", required=True, help="Path to second signature image")
    parser.add_argument("--checkpoint", default=None, help="Path to encoder checkpoint")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: from training)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    encoder = load_encoder(args.checkpoint, device)

    # Verify
    result = verify_signatures(
        encoder, args.img1, args.img2,
        threshold=args.threshold, device=device
    )


    # Display results
    print(f"\n{'='*50}")
    print(f"Signature Verification Result")
    print(f"{'='*50}")
    print(f"Image 1: {args.img1}")
    print(f"Image 2: {args.img2}")
    print(f"{'-'*50}")
    print(f"Euclidean Distance : {result['distance']:.4f}")
    print(f"Cosine Similarity  : {result['similarity']:.4f}")
    print(f"Threshold          : {result['threshold']:.4f}")
    print(f"Confidence         : {result['confidence']:.1%}")
    print(f"{'-'*50}")

    if result["is_match"]:
        print(f">>> MATCH -- Signatures likely belong to the SAME person")
    else:
        print(f">>> NO MATCH -- Signatures likely belong to DIFFERENT people")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
