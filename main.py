"""
Signature Verification System — CLI Entry Point

Usage:
    python main.py train [--epochs 50] [--batch-size 32] [--lr 1e-4]
                         [--mlflow-tracking-uri http://localhost:5000]
                         [--experiment-name signature-verification]
                         [--run-name my-run]
                         [--register-model SignatureVerificationEncoder]
                         [--no-mlflow]
    python main.py verify --img1 path/to/sig1.png --img2 path/to/sig2.png
                          [--checkpoint checkpoints/best_encoder.pth]
                          [--model-uri models:/SignatureVerificationEncoder/latest]
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Signature Verification System with MLflow Tracking")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python main.py train   [training & mlflow options]")
        print("  python main.py verify  --img1 <path> --img2 <path> [options]")
        print()
        print("Commands:")
        print("  train   Train the Siamese signature encoder with MLflow logging")
        print("  verify  Verify two signature images (local checkpoint or MLflow URI)")
        print()
        print("MLflow Tracking UI (Docker):")
        print("  Start tracking server:  docker compose up -d")
        print("  Access web interface:   http://localhost:5000")
        print("  Stop tracking server:   docker compose down")
        print()
        print("Examples:")
        print("  python main.py train --epochs 50 --batch-size 32")
        print("  python main.py train --epochs 10 --register-model SignatureVerificationEncoder")
        print("  python main.py verify --img1 ref1.png --img2 ref2.png")
        print("  python main.py verify --img1 ref1.png --img2 ref2.png --model-uri models:/SignatureVerificationEncoder/latest")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)

    command = sys.argv[1]

    # Remove the command from argv so argparse in submodules works correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from train import main as train_main
        train_main()
    elif command == "verify":
        from verify import main as verify_main
        verify_main()
    else:
        print(f"Unknown command: {command}")
        print("Use 'train' or 'verify'")
        sys.exit(1)


if __name__ == "__main__":
    main()