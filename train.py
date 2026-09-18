"""
train.py — Training loop for the Siamese signature encoder with MLflow tracking.

Usage:
    python train.py [--epochs 50] [--batch-size 32] [--lr 1e-4] [--pairs 1000]
                    [--mlflow-tracking-uri http://localhost:5000]
                    [--experiment-name signature-verification]
                    [--run-name my-experiment]
                    [--register-model SignatureVerificationEncoder]

Trains on users {u014, u019, u022, u028}, validates on {u07, u09}.
Saves the best encoder checkpoint, loss curves, biometric metrics,
and registers artifacts to MLflow.
"""

import os
import argparse
import time
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader

from dataset import SignaturePairDataset
from model import SiameseNetwork, ContrastiveLoss
from mlflow_utils import MLflowTracker


# --- Configuration ---
DATASET_ROOT = os.path.join("Dataset", "Dataset")
CHECKPOINT_DIR = "checkpoints"
TRAIN_USERS = ["u014", "u019", "u022", "u028"]
VAL_USERS = ["u07", "u09"]


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for img1, img2, labels in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, criterion, dataloader, device):
    """Run validation and return average loss."""
    model.eval()
    running_loss = 0.0
    num_batches = 0

    for img1, img2, labels in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def compute_metrics(model, dataloader, device):
    """Compute distances and labels for ROC analysis."""
    model.eval()
    all_distances = []
    all_labels = []

    for img1, img2, labels in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        emb1, emb2 = model(img1, img2)
        distances = torch.nn.functional.pairwise_distance(emb1, emb2, p=2)

        all_distances.extend(distances.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_distances), np.array(all_labels)


def find_eer(distances, labels):
    """
    Find the Equal Error Rate (EER) — the threshold where FAR == FRR.

    For positive pairs (label=1), a large distance is a false rejection.
    For negative pairs (label=0), a small distance is a false acceptance.
    """
    thresholds = np.linspace(0, 2.0, 500)
    best_diff = float("inf")
    eer_threshold = 0.0
    eer_value = 0.0

    for t in thresholds:
        # False Acceptance Rate: negative pairs accepted (distance < threshold)
        neg_mask = labels == 0
        far = np.mean(distances[neg_mask] < t) if neg_mask.any() else 0.0

        # False Rejection Rate: positive pairs rejected (distance >= threshold)
        pos_mask = labels == 1
        frr = np.mean(distances[pos_mask] >= t) if pos_mask.any() else 0.0

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer_threshold = t
            eer_value = (far + frr) / 2

    return eer_value, eer_threshold


def plot_loss_curves(train_losses, val_losses, save_path):
    """Save training and validation loss curves."""
    fig = plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Contrastive Loss", fontsize=12)
    plt.title("Siamese Encoder Training Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {save_path}")


def plot_distance_distribution(distances, labels, threshold, save_path):
    """Plot distribution of distances for positive/negative pairs."""
    fig = plt.figure(figsize=(10, 6))

    pos_dist = distances[labels == 1]
    neg_dist = distances[labels == 0]

    plt.hist(pos_dist, bins=30, alpha=0.6, color="green", label="Genuine pairs", density=True)
    plt.hist(neg_dist, bins=30, alpha=0.6, color="red", label="Negative pairs", density=True)
    plt.axvline(x=threshold, color="blue", linestyle="--", linewidth=2,
                label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Euclidean Distance", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distance Distribution (Validation Set)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Distance distribution saved to {save_path}")


def plot_roc_curve(distances, labels, eer_threshold, eer_value, save_path):
    """Plot Receiver Operating Characteristic (ROC) curve with EER marked."""
    # Scores: smaller distance = more similar (positive class)
    scores = -distances
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance")

    # Mark EER operating point (where FAR ~ FRR, i.e., FPR ~ 1 - TPR)
    plt.scatter([eer_value], [1.0 - eer_value], color="red", s=80, zorder=5,
                label=f"EER = {eer_value:.4f} (Thresh={eer_threshold:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to {save_path}")
    return float(roc_auc)


def main():
    parser = argparse.ArgumentParser(description="Train Siamese Signature Encoder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--pairs", type=int, default=1000, help="Pairs per epoch")
    parser.add_argument("--margin", type=float, default=1.0, help="Contrastive loss margin")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Optimizer weight decay")
    # MLflow arguments
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking server URI (default: http://localhost:5000 or env MLFLOW_TRACKING_URI)")
    parser.add_argument("--experiment-name", type=str, default="signature-verification",
                        help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--register-model", type=str, default=None,
                        help="Register the trained model in MLflow Model Registry under this name")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize MLflow Tracker
    tracker = MLflowTracker(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        enabled=not args.no_mlflow,
    )

    # --- Datasets ---
    print(f"\nTraining users: {TRAIN_USERS}")
    print(f"Validation users: {VAL_USERS}")

    train_dataset = SignaturePairDataset(
        DATASET_ROOT, user_ids=TRAIN_USERS,
        pairs_per_epoch=args.pairs, augment=True, seed=42
    )
    val_dataset = SignaturePairDataset(
        DATASET_ROOT, user_ids=VAL_USERS,
        pairs_per_epoch=args.pairs // 2, augment=False, seed=123
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )

    print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

    # --- Model ---
    model = SiameseNetwork(
        embedding_dim=args.embedding_dim, pretrained=True
    ).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Start MLflow run
    tracker.start_run(tags={
        "model_type": "SiameseNetwork",
        "backbone": "ResNet-18",
        "dataset": "sigverify_internal",
    })

    # Log initial hyperparameters and configuration
    tracker.log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "pairs_per_epoch": args.pairs,
        "contrastive_margin": args.margin,
        "patience": args.patience,
        "embedding_dim": args.embedding_dim,
        "weight_decay": args.weight_decay,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "train_users": ",".join(TRAIN_USERS),
        "val_users": ",".join(VAL_USERS),
        "train_pairs_count": len(train_dataset),
        "val_pairs_count": len(val_dataset),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "image_height": 155,
        "image_width": 220,
    })

    # --- Training Loop ---
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, lr={args.lr}, margin={args.margin}")
    print(f"{'='*60}\n")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            # Re-mine pairs each epoch for diversity
            train_dataset.reshuffle()

            train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
            val_loss = validate(model, criterion, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            # Log metrics to MLflow
            tracker.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
                "epoch_duration_sec": elapsed,
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_encoder.pth")
                torch.save({
                    "epoch": epoch,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "embedding_dim": args.embedding_dim,
                }, checkpoint_path)
                print(f"  [*] Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

        # --- Post-training Analysis ---
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}")

        # Plot and log loss curves
        loss_curve_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        tracker.log_artifact(loss_curve_path)

        # Load best model for biometric evaluation
        best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_encoder.pth")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(
                best_checkpoint_path,
                map_location=device, weights_only=True
            )
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            tracker.log_artifact(best_checkpoint_path, artifact_path="checkpoints")

        # Compute EER and biometric metrics on validation set
        val_dataset_eval = SignaturePairDataset(
            DATASET_ROOT, user_ids=VAL_USERS,
            pairs_per_epoch=max(500, args.pairs // 2), augment=False, seed=999
        )
        eval_loader = DataLoader(val_dataset_eval, batch_size=args.batch_size, shuffle=False)

        distances, labels = compute_metrics(model, eval_loader, device)
        eer, eer_threshold = find_eer(distances, labels)

        # Distance statistics
        pos_dist = distances[labels == 1]
        neg_dist = distances[labels == 0]

        # Accuracy at optimal threshold
        predictions = (distances < eer_threshold).astype(float)
        accuracy = float((predictions == labels).mean())

        # Plot distance distribution
        dist_plot_path = os.path.join(CHECKPOINT_DIR, "distance_distribution.png")
        plot_distance_distribution(distances, labels, eer_threshold, dist_plot_path)
        tracker.log_artifact(dist_plot_path)

        # Plot ROC curve and compute AUC
        roc_plot_path = os.path.join(CHECKPOINT_DIR, "roc_curve.png")
        roc_auc = plot_roc_curve(distances, labels, eer_threshold, eer, roc_plot_path)
        tracker.log_artifact(roc_plot_path)

        # Save threshold locally
        threshold_path = os.path.join(CHECKPOINT_DIR, "threshold.pth")
        torch.save(
            {"threshold": float(eer_threshold), "eer": float(eer), "accuracy": accuracy, "auc": roc_auc},
            threshold_path
        )
        tracker.log_artifact(threshold_path, artifact_path="checkpoints")

        # Summary dictionary artifact
        summary = {
            "best_val_loss": float(best_val_loss),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "accuracy_at_eer": float(accuracy),
            "roc_auc": float(roc_auc),
            "genuine_pairs_mean_distance": float(np.mean(pos_dist)) if len(pos_dist) > 0 else 0.0,
            "genuine_pairs_std_distance": float(np.std(pos_dist)) if len(pos_dist) > 0 else 0.0,
            "negative_pairs_mean_distance": float(np.mean(neg_dist)) if len(neg_dist) > 0 else 0.0,
            "negative_pairs_std_distance": float(np.std(neg_dist)) if len(neg_dist) > 0 else 0.0,
        }
        summary_path = os.path.join(CHECKPOINT_DIR, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        tracker.log_artifact(summary_path)

        # Log final evaluation metrics
        tracker.log_metrics({
            "best_val_loss": float(best_val_loss),
            "val_eer": float(eer),
            "val_eer_threshold": float(eer_threshold),
            "val_accuracy_at_eer": float(accuracy),
            "val_roc_auc": float(roc_auc),
            "val_genuine_dist_mean": summary["genuine_pairs_mean_distance"],
            "val_negative_dist_mean": summary["negative_pairs_mean_distance"],
        })

        print(f"\nValidation Metrics:")
        print(f"  EER: {eer:.4f} ({eer*100:.1f}%)")
        print(f"  Optimal threshold: {eer_threshold:.4f}")
        print(f"  Accuracy at EER threshold: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  ROC AUC: {roc_auc:.4f}")

        # Log PyTorch Model to MLflow
        dummy_input = torch.randn(1, 1, 155, 220, device=device)
        tracker.log_model(
            model=model.encoder,
            artifact_path="model",
            input_example=dummy_input,
            registered_model_name=args.register_model,
        )

        tracker.end_run(status="FINISHED")

    except Exception as e:
        print(f"\nError during training: {e}")
        tracker.end_run(status="FAILED")
        raise e


if __name__ == "__main__":
    main()
