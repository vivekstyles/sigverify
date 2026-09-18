"""
verify.py — Signature verification using trained Siamese encoder with MLflow support.

Usage:
    python verify.py --img1 path/to/sig1.png --img2 path/to/sig2.png
    python verify.py --img1 path/to/sig1.png --img2 path/to/sig2.png --threshold 0.5
    python verify.py --img1 path/to/sig1.png --img2 path/to/sig2.png --model-uri models:/SignatureVerificationEncoder/latest

Computes embedding distance between two signature images and outputs
a similarity score with an accept/reject decision.
"""

import os
import sys

# Ensure UTF-8 stdout/stderr on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import SignatureEncoder

# Suppress noisy external warnings
logger = logging.getLogger("sigverify.verify")


# Constants
IMG_HEIGHT = 155
IMG_WIDTH = 220
CHECKPOINT_DIR = "checkpoints"


def load_encoder(checkpoint_path=None, model_uri=None, tracking_uri=None, device="cpu"):
    """
    Load trained encoder from either an MLflow model URI or a local checkpoint.

    Args:
        checkpoint_path: Path to local .pth checkpoint.
        model_uri: MLflow model URI (e.g., 'models:/SignatureVerificationEncoder/latest'
                   or 'runs:/<run_id>/model').
        tracking_uri: MLflow tracking server URI (defaults to env or http://localhost:5000).
        device: Device to map model to.

    Returns:
        Loaded SignatureEncoder in eval mode.
    """
    if model_uri:
        import mlflow
        import mlflow.pytorch
        effective_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(effective_tracking_uri)
        print(f"Connecting to MLflow Tracking Server at: {effective_tracking_uri}")
        print(f"Loading model from MLflow URI: {model_uri}...")
        encoder = mlflow.pytorch.load_model(model_uri, map_location=device)
        if hasattr(encoder, "to"):
            try:
                encoder.to(device)
            except Exception:
                pass
        if hasattr(encoder, "eval"):
            try:
                encoder.eval()
            except (NotImplementedError, AttributeError):
                pass
        return encoder

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_encoder.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to train the encoder, or supply --model-uri."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    embedding_dim = checkpoint.get("embedding_dim", 128)

    encoder = SignatureEncoder(embedding_dim=embedding_dim, pretrained=False)
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
    try:
        embedding = encoder(img_tensor)
    except Exception:
        if hasattr(encoder, "module"):
            embedding = encoder.module()(img_tensor)
        else:
            raise
    if isinstance(embedding, (tuple, list)):
        embedding = embedding[0]
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
    parser.add_argument("--model-uri", default=None,
                        help="MLflow model URI (e.g. models:/SignatureVerificationEncoder/latest or runs:/<run_id>/model)")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                        help="MLflow tracking server URI (default: http://localhost:5000 or env MLFLOW_TRACKING_URI)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: from training)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    encoder = load_encoder(
        checkpoint_path=args.checkpoint,
        model_uri=args.model_uri,
        tracking_uri=args.mlflow_tracking_uri,
        device=device
    )

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
