"""
Signature Verification System — CLI Entry Point

Usage:
    python main.py train [--epochs 50] [--batch-size 32] [--lr 1e-4]
    python main.py verify --img1 path/to/sig1.png --img2 path/to/sig2.png
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Signature Verification System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python main.py train   [training options]")
        print("  python main.py verify  --img1 <path> --img2 <path>")
        print()
        print("Commands:")
        print("  train   Train the Siamese signature encoder")
        print("  verify  Verify two signature images")
        print()
        print("Examples:")
        print("  python main.py train --epochs 50 --batch-size 32")
        print("  python main.py verify --img1 sig_a.png --img2 sig_b.png")
        sys.exit(1)

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