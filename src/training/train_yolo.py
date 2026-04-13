"""
Fase 1 — Training YOLOv8 per Window Detection

Addestra YOLOv8 a rilevare finestre in facciate di edifici.
Output: bounding box (x, y, w, h) per ogni finestra trovata nell'immagine.

Task: Object Detection (NON segmentazione — nessuna maschera pixel)
Dataset: window-detection-vnpow-v1 (Roboflow, formato COCO)
Modello: yolov8n pretrained su COCO → fine-tuning sul nostro dataset

Esecuzione:
    # Smoke test (1 epoca, verifica che tutto funzioni)
    python src/training/train_yolo.py --epochs 1

    # Training completo
    python src/training/train_yolo.py

    # Training con modello più grande
    python src/training/train_yolo.py --model s --epochs 100

    # Background sulla VM
    nohup python src/training/train_yolo.py > runs/train_yolo.log 2>&1 &
"""

import argparse
import os
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Percorsi — tutti relativi alla root del progetto (dove si esegue lo script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_YAML = PROJECT_ROOT / "data" / "window_detection.yaml"
CONFIG_TRAINING = PROJECT_ROOT / "configs" / "training.yaml"
CONFIG_MODEL = PROJECT_ROOT / "configs" / "model.yaml"


def load_config() -> dict:
    with open(CONFIG_TRAINING) as f:
        return yaml.safe_load(f)["yolo"]


def run_training(args: argparse.Namespace) -> None:
    cfg = load_config()

    # Parametri: CLI sovrascrive config yaml
    model_variant = args.model or cfg.get("architecture", "n").replace("yolov8", "")
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch or cfg["batch_size"]
    device = args.device or str(cfg["device"])
    img_size = cfg["img_size"]
    project = str(PROJECT_ROOT / cfg["project"])
    name = cfg["name"]

    weights = f"yolov8{model_variant}.pt"

    # Verifica che il dataset esista
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML non trovato: {DATASET_YAML}")

    print(f"\n{'='*60}")
    print(f"  YOLOv8{model_variant.upper()} — Window Detection Training")
    print(f"{'='*60}")
    print(f"  Dataset:  {DATASET_YAML}")
    print(f"  Weights:  {weights} (pretrained COCO → fine-tuning)")
    print(f"  Epochs:   {epochs}")
    print(f"  Batch:    {batch_size}")
    print(f"  Img size: {img_size}px")
    print(f"  Device:   {device}")
    print(f"  Output:   {project}/{name}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # MLflow tracking
    # ------------------------------------------------------------------
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("window-detection-yolov8")

    with mlflow.start_run(run_name=f"yolov8{model_variant}_ep{epochs}"):
        # Log parametri
        mlflow.log_params({
            "model": f"yolov8{model_variant}",
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "device": device,
            "dataset": "window-detection-vnpow-v1",
        })

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        model = YOLO(weights)

        results = model.train(
            data=str(DATASET_YAML),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            workers=cfg.get("workers", 4),
            seed=cfg.get("random_seed", 42),
            patience=cfg.get("patience", 20),
            # Optimizer
            optimizer=cfg.get("optimizer", "SGD"),
            lr0=cfg.get("lr0", 0.01),
            lrf=cfg.get("lrf", 0.01),
            momentum=cfg.get("momentum", 0.937),
            weight_decay=cfg.get("weight_decay", 0.0005),
            warmup_epochs=cfg.get("warmup_epochs", 3),
            cos_lr=cfg.get("cos_lr", True),
            # Mixed precision
            amp=cfg.get("amp", True),
            # Output
            project=project,
            name=name,
            exist_ok=cfg.get("exist_ok", True),
            # Augmentation (da config yaml tramite ultralytics hyperparams)
            hsv_h=cfg.get("augmentation", {}).get("hsv_h", 0.015),
            hsv_s=cfg.get("augmentation", {}).get("hsv_s", 0.7),
            hsv_v=cfg.get("augmentation", {}).get("hsv_v", 0.4),
            degrees=cfg.get("augmentation", {}).get("degrees", 5.0),
            translate=cfg.get("augmentation", {}).get("translate", 0.1),
            scale=cfg.get("augmentation", {}).get("scale", 0.5),
            flipud=cfg.get("augmentation", {}).get("flipud", 0.0),
            fliplr=cfg.get("augmentation", {}).get("fliplr", 0.5),
            mosaic=cfg.get("augmentation", {}).get("mosaic", 1.0),
            mixup=cfg.get("augmentation", {}).get("mixup", 0.0),
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Log metriche finali su MLflow
        # ------------------------------------------------------------------
        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50":      metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95":   metrics.get("metrics/mAP50-95(B)", 0),
            "precision":  metrics.get("metrics/precision(B)", 0),
            "recall":     metrics.get("metrics/recall(B)", 0),
        })

        # Log artefatti: best weights + confusion matrix
        best_weights = Path(project) / name / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            print(f"\n✅ Best weights salvati: {best_weights}")

        results_dir = Path(project) / name
        for artifact in ["confusion_matrix.png", "results.png", "PR_curve.png"]:
            artifact_path = results_dir / artifact
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="plots")

        print(f"\n✅ Training completato!")
        print(f"   mAP50:    {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   MLflow:   {mlflow_uri}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training YOLOv8 per Window Detection"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["n", "s", "m", "l", "x"],
        help="Variante YOLOv8: n=nano, s=small, m=medium (default: da config yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Numero di epoche (default: da config yaml)"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: da config yaml)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: '0' per GPU, 'cpu' per CPU (default: da config yaml)"
    )
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
