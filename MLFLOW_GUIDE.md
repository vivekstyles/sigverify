# Production MLflow Guide for Signature Verification

This guide explains how to use the production-grade MLflow tracking server, Dockerized UI, experiment tracking, artifact store, and Model Registry in this project.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MLflow Tracking UI                            │
│                     (http://localhost:5000 in Browser)                  │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │   Docker Container (mlflow)   │
                     │  - MLflow Server (Port 5000)  │
                     │  - SQLite Backend Store       │
                     │  - Artifact Store Directory   │
                     └───────────────┬───────────────┘
                                     │ (Volume Mount)
                     ┌───────────────▼───────────────┐
                     │      Host: ./mlflow_data      │
                     │  ├── data/mlflow.db           │
                     │  └── artifacts/               │
                     └───────────────────────────────┘
                                     ▲
                                     │ (HTTP REST API or local fallback)
                     ┌───────────────┴───────────────┐
                     │     Python Client Pipeline    │
                     │  - train.py (Metrics, Params) │
                     │  - verify.py (Inference/Eval) │
                     │  - mlflow_utils.py (Tracker)  │
                     └───────────────────────────────┘
```

### Key Production Features
1. **Containerized UI & Server**: Zero setup on the host; isolated Docker container running the MLflow tracking server with automated health checks.
2. **Persistent Storage**: All runs, metrics, artifacts, and registered models persist on host disk at `./mlflow_data`.
3. **Graceful Fallback & Fault Tolerance**: If the Docker server is offline or unreachable, `mlflow_utils.py` automatically falls back to local file tracking (`./mlruns`) with a warning instead of halting training.
4. **Biometric Metrics & Visualizations**: Automatically logs Equal Error Rate (EER), optimal thresholds, ROC curves with AUC, distance distributions, and contrastive loss curves.
5. **Model Registry & Model Serving**: Saves the PyTorch `SignatureEncoder` with tensor schema signatures and supports loading models by MLflow URI (`models:/<name>/<version>`).

---

## 2. Quick Start: Docker MLflow UI

### Start the Tracking Server
Run from the project root:
```bash
docker compose up -d
```

### Open the Web Interface
Navigate to **[http://localhost:5000](http://localhost:5000)** in your browser.

### Inspect Container Status & Health
```bash
# Check container status (should show 'healthy')
docker compose ps

# View live container logs
docker compose logs -f mlflow
```

### Stop the Server
```bash
# Stop containers (data is preserved in ./mlflow_data)
docker compose down

# Stop and remove all volumes (WARNING: clears database & artifacts)
docker compose down -v
```

---

## 3. Training with MLflow

### Standard Training Run
```bash
python main.py train --epochs 50 --batch-size 32
```

### Training with Custom Run Name & Experiment
```bash
python main.py train \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --experiment-name "signature-verification" \
    --run-name "resnet18-baseline"
```

### Register Model into MLflow Model Registry
To register the resulting model directly into the registry for production deployment:
```bash
python main.py train \
    --epochs 50 \
    --register-model "SignatureVerificationEncoder"
```

### Offline / Standalone Mode (Disable MLflow)
If you wish to train without logging to any MLflow tracking server:
```bash
python main.py train --epochs 10 --no-mlflow
```

---

## 4. What Gets Tracked

### Parameters
- **Hyperparameters**: `epochs`, `batch_size`, `learning_rate`, `pairs_per_epoch`, `contrastive_margin`, `patience`, `embedding_dim`, `weight_decay`, `optimizer`, `scheduler`.
- **Dataset Configuration**: `train_users`, `val_users`, `train_pairs_count`, `val_pairs_count`, `image_height`, `image_width`.
- **Architecture**: `backbone` (ResNet-18), `total_parameters`, `trainable_parameters`.

### Metrics (Per-Epoch & Final)
- `train_loss`, `val_loss`, `learning_rate`, `epoch_duration_sec` (logged each epoch with step tracking).
- `best_val_loss`: Lowest validation loss achieved.
- `val_eer`: Equal Error Rate on validation pairs.
- `val_eer_threshold`: Optimal distance threshold where FAR equals FRR.
- `val_accuracy_at_eer`: Classification accuracy at the optimal threshold.
- `val_roc_auc`: Area Under the ROC Curve.
- `val_genuine_dist_mean`, `val_negative_dist_mean`: Mean embedding distances for genuine and negative pairs.

### System & Environment Tags
- `git_commit`: Current git commit SHA.
- `git_branch`: Current git branch name.
- `torch_version`, `python_version`, `os`, `device_type`, `cuda_device_name`.

### Logged Artifacts
- `loss_curve.png`: Contrastive loss progression across epochs.
- `distance_distribution.png`: Histogram of genuine vs. forged Euclidean distances with threshold marker.
- `roc_curve.png`: Full Receiver Operating Characteristic curve with marked EER operating point.
- `evaluation_summary.json`: Machine-readable summary of all biometric metrics and distance statistics.
- `checkpoints/best_encoder.pth`: PyTorch state dictionary checkpoint.
- `checkpoints/threshold.pth`: Serialized optimal decision threshold.
- `model/`: Registered PyTorch model with schema signature and conda/pip dependencies.

---

## 5. Verification & Inference using MLflow Models

You can verify signatures using either a local `.pth` checkpoint or an MLflow Model URI:

### Using Local Checkpoint
```bash
python main.py verify \
    --img1 ref1.png \
    --img2 ref2.png \
    --checkpoint checkpoints/best_encoder.pth
```

### Using MLflow Model Registry URI
```bash
# Load latest version from Model Registry
python main.py verify \
    --img1 ref1.png \
    --img2 ref2.png \
    --model-uri "models:/SignatureVerificationEncoder/latest"

# Or load specific version (e.g., version 1)
python main.py verify \
    --img1 ref1.png \
    --img2 ref2.png \
    --model-uri "models:/SignatureVerificationEncoder/1"
```

### Using MLflow Run Artifact URI
```bash
python main.py verify \
    --img1 ref1.png \
    --img2 ref2.png \
    --model-uri "runs:/<RUN_ID>/model"
```

---

## 6. Managing the Model Registry

In the MLflow UI (**http://localhost:5000**):
1. Navigate to the **Models** tab in the top navigation.
2. Click on **`SignatureVerificationEncoder`**.
3. View registered versions, their source runs, parameters, and metrics.
4. Add tags (e.g. `stage: production`, `validated: true`) or aliases (e.g. `champion`, `challenger`) to manage model deployment stages.

---

## 7. Storage and Backups

The SQLite database and all artifacts reside on the host in:
```
./mlflow_data/
├── data/
│   └── mlflow.db         # SQLite metadata database
└── artifacts/            # Model weights, plots, and checkpoints
```

To back up your tracking history, simply back up the `./mlflow_data` directory.
