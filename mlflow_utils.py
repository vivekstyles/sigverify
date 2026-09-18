"""
mlflow_utils.py — Production MLflow Tracking & Lifecycle Management.

Provides robust experiment tracking, artifact logging, model registry
integration, system metadata extraction, and fault-tolerant fallback
if the tracking server is temporarily unreachable.
"""

import os
import sys

# Ensure UTF-8 stdout/stderr on Windows consoles to prevent UnicodeEncodeError with MLflow emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import platform
import subprocess
import urllib.request
import logging
import json
from typing import Dict, Any, Optional

import torch
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

logger = logging.getLogger("sigverify.mlflow")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[MLflow] %(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def check_server_health(tracking_uri: str, timeout: float = 3.0) -> bool:
    """Check if remote MLflow server is responsive."""
    if not tracking_uri.startswith(("http://", "https://")):
        # Local file or sqlite path, always valid
        return True

    health_url = tracking_uri.rstrip("/") + "/health"
    try:
        req = urllib.request.Request(
            health_url,
            headers={"User-Agent": "SigVerify-MLflow-Client/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception as e:
        # Fallback check on root URL if /health is not implemented
        try:
            req = urllib.request.Request(
                tracking_uri,
                headers={"User-Agent": "SigVerify-MLflow-Client/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status in (200, 301, 302)
        except Exception:
            return False


def get_git_metadata() -> Dict[str, str]:
    """Retrieve Git commit and branch information safely."""
    meta = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        meta["git_branch"] = branch
    except Exception:
        meta["git_branch"] = "unknown"

    return meta


def get_system_metadata() -> Dict[str, str]:
    """Capture hardware, OS, and environment information."""
    meta = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        meta["cuda_device_count"] = str(torch.cuda.device_count())
    return meta


class MLflowTracker:
    """
    Production wrapper for MLflow experiment tracking.

    Handles:
      - Automatic server discovery and graceful local fallback
      - System and Git tag injection
      - Structured parameter & metric logging
      - Plot, checkpoint, and model registry artifacts
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "signature-verification",
        run_name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run = None
        self.experiment_name = experiment_name
        self.run_name = run_name

        if not self.enabled:
            logger.info("MLflow tracking is disabled by configuration.")
            return

        # Determine tracking URI: parameter > env var > default
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        # Verify server availability if remote
        if uri.startswith(("http://", "https://")):
            if not check_server_health(uri):
                logger.warning(
                    f"MLflow server at {uri} is not reachable. "
                    "Falling back to local file tracking at './mlruns'."
                )
                uri = "file:./mlruns"

        self.tracking_uri = uri
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(
            f"Initialized MLflow tracker (URI: {self.tracking_uri}, Experiment: '{self.experiment_name}')"
        )

    def start_run(self, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run with system and git tags."""
        if not self.enabled:
            return None

        combined_tags = {}
        combined_tags.update(get_git_metadata())
        combined_tags.update(get_system_metadata())
        if tags:
            combined_tags.update(tags)

        self.run = mlflow.start_run(run_name=self.run_name, tags=combined_tags)
        logger.info(f"Started MLflow run: ID={self.run.info.run_id}, Name={self.run_name or 'unnamed'}")
        return self.run

    def log_params(self, params: Dict[str, Any]):
        """Log hyper-parameters, casting complex types to strings."""
        if not self.enabled:
            return

        sanitized = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple, dict, set)):
                sanitized[k] = str(v)
            elif v is None:
                sanitized[k] = "None"
            else:
                sanitized[k] = v
        mlflow.log_params(sanitized)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for the current step/epoch."""
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file or directory as an MLflow artifact."""
        if not self.enabled:
            return
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        else:
            logger.warning(f"Artifact path does not exist, skipping: {local_path}")

    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure directly."""
        if not self.enabled:
            return
        mlflow.log_figure(figure, artifact_file)
        logger.info(f"Logged figure artifact: {artifact_file}")

    def log_dict(self, data: Dict[str, Any], artifact_file: str):
        """Log a dictionary directly as a JSON artifact."""
        if not self.enabled:
            return
        mlflow.log_dict(data, artifact_file)
        logger.info(f"Logged JSON dictionary: {artifact_file}")

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        input_example: Optional[torch.Tensor] = None,
        registered_model_name: Optional[str] = None,
    ):
        """
        Log PyTorch model with schema signature and optional model registry registration.
        """
        if not self.enabled:
            return

        signature = None
        if input_example is not None:
            try:
                # Infer model signature: input tensor -> output tensor
                with torch.no_grad():
                    model.eval()
                    output_example = model(input_example)
                signature = infer_signature(
                    input_example.cpu().numpy(),
                    output_example.cpu().numpy()
                )
            except Exception as e:
                logger.warning(f"Could not infer model signature: {e}")

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example.cpu().numpy() if input_example is not None else None,
            registered_model_name=registered_model_name,
        )
        logger.info(
            f"PyTorch model successfully logged at '{artifact_path}'"
            + (f" and registered as '{registered_model_name}'" if registered_model_name else "")
        )
        return model_info

    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run."""
        if not self.enabled or self.run is None:
            return
        try:
            mlflow.end_run(status=status)
        except UnicodeEncodeError:
            # Fallback if Windows terminal cannot render the MLflow runner emoji
            pass
        logger.info(f"Completed MLflow run (Status: {status})")
        self.run = None

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
