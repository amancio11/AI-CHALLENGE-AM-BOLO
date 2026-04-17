"""
train_efficientnet.py

Addestra un classificatore EfficientNet-B2 (timm, pretrained ImageNet)
per classificare finestre come clean/dirty.

Struttura attesa del dataset:
    data/windows_efficientnet/
        train/clean/  train/dirty/
        valid/clean/  valid/dirty/
        test/clean/   test/dirty/

Training in due fasi:
    Fase 1 (warmup)  — backbone frozen, solo il classifier head viene aggiornato
    Fase 2 (finetune)— tutti i layer sbloccati, learning rate più basso

Metriche loggate su MLflow: loss, accuracy, F1, precision, recall per ogni epoch.
Modello migliore (val F1-dirty) salvato in models/weights/efficientnet_best.pt

Uso sulla VM:
    conda activate glass-cv
    cd /mnt/project/AI-CHALLENGE-AM-BOLO
    python src/training/train_efficientnet.py

    # Opzioni:
    python src/training/train_efficientnet.py --epochs1 5 --epochs2 30 --batch 32 --lr 3e-4
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Percorsi ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "windows_efficientnet"
WEIGHTS_DIR = ROOT / "models" / "weights"

MODEL_NAME   = "efficientnet_b2"
NUM_CLASSES  = 2          # 0=clean, 1=dirty
IMAGE_SIZE   = 260        # input nativo di EfficientNet-B2
DIRTY_IDX    = 1          # indice della classe dirty nel dataset (ordine alfabetico: clean=0, dirty=1)


# ── Trasformazioni ────────────────────────────────────────────────────────────
def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


# ── Costruzione modello ────────────────────────────────────────────────────────
def build_model(freeze_backbone: bool) -> nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Congela tutto tranne il classifier head (ultimo layer)
            if "classifier" not in name:
                param.requires_grad = False
    return model


# ── Un epoch di training ────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ── Valutazione ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1_dirty = f1_score(all_labels, all_preds, pos_label=DIRTY_IDX, average="binary", zero_division=0)
    return total_loss / total, correct / total, f1_dirty, all_preds, all_labels


# ── Loop di training ────────────────────────────────────────────────────────────
def train_phase(
    model, train_loader, valid_loader, criterion, device,
    epochs, lr, phase_name, best_f1, best_model_path
):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"[{phase_name}] epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  val_f1_dirty={val_f1:.3f}  "
            f"({elapsed:.0f}s)"
        )

        mlflow.log_metrics({
            f"{phase_name}/train_loss": train_loss,
            f"{phase_name}/train_acc": train_acc,
            f"{phase_name}/val_loss": val_loss,
            f"{phase_name}/val_acc": val_acc,
            f"{phase_name}/val_f1_dirty": val_f1,
        }, step=epoch)

        # Salva il miglior modello in base a val F1-dirty
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"    ↑ Nuovo best model salvato (val_f1_dirty={val_f1:.4f})")

    return best_f1


# ── Main ────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    for split in ("train", "valid", "test"):
        p = DATA_DIR / split
        if not p.exists():
            raise FileNotFoundError(
                f"Cartella non trovata: {p}\n"
                f"Esegui prima: python src/data/build_efficientnet_dataset.py"
            )

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=get_transforms(train=True))
    valid_ds = datasets.ImageFolder(DATA_DIR / "valid", transform=get_transforms(train=False))
    test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=get_transforms(train=False))

    print(f"[+] Dataset: train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}")
    print(f"    Classi: {train_ds.class_to_idx}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = WEIGHTS_DIR / "efficientnet_best.pt"

    # Calcolo class weights per gestire eventuale sbilanciamento
    counts = np.array([len(list((DATA_DIR / "train" / c).iterdir())) for c in ("clean", "dirty")])
    weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    weights = weights / weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights)

    # MLflow
    mlflow.set_experiment("efficientnet_cleanliness")
    run_name = f"effnetb2_ep{args.epochs1 + args.epochs2}_b{args.batch}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "batch_size": args.batch,
            "epochs_warmup": args.epochs1,
            "epochs_finetune": args.epochs2,
            "lr_warmup": args.lr,
            "lr_finetune": args.lr / 10,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "n_train": len(train_ds),
            "n_valid": len(valid_ds),
            "n_test": len(test_ds),
        })

        best_f1 = 0.0

        # ── Fase 1: warmup (backbone frozen) ───────────────────────────────────
        print(f"\n{'='*60}")
        print(f"FASE 1 — Warmup ({args.epochs1} epoche, backbone frozen)")
        print(f"{'='*60}")
        model = build_model(freeze_backbone=True).to(device)
        best_f1 = train_phase(
            model, train_loader, valid_loader, criterion, device,
            epochs=args.epochs1, lr=args.lr,
            phase_name="warmup", best_f1=best_f1, best_model_path=best_model_path
        )

        # ── Fase 2: fine-tuning (tutti i layer sbloccati) ─────────────────────
        print(f"\n{'='*60}")
        print(f"FASE 2 — Fine-tuning ({args.epochs2} epoche, tutti i layer)")
        print(f"{'='*60}")
        for param in model.parameters():
            param.requires_grad = True
        best_f1 = train_phase(
            model, train_loader, valid_loader, criterion, device,
            epochs=args.epochs2, lr=args.lr / 10,
            phase_name="finetune", best_f1=best_f1, best_model_path=best_model_path
        )

        # ── Valutazione finale sul test set ───────────────────────────────────
        print(f"\n{'='*60}")
        print("VALUTAZIONE FINALE — test set")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        _, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion, device)

        report = classification_report(labels, preds, target_names=["clean", "dirty"], digits=3)
        print(report)

        mlflow.log_metrics({"test/accuracy": test_acc, "test/f1_dirty": test_f1})
        mlflow.log_text(report, "test_classification_report.txt")
        mlflow.log_artifact(str(best_model_path))

        print(f"\n[✓] Modello migliore salvato in: {best_model_path}")
        print(f"[✓] Test accuracy: {test_acc:.4f}  |  F1-dirty: {test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B2 per classificazione clean/dirty")
    parser.add_argument("--epochs1", type=int, default=5,
                        help="Epoche warmup con backbone frozen (default: 5)")
    parser.add_argument("--epochs2", type=int, default=30,
                        help="Epoche fine-tuning completo (default: 30)")
    parser.add_argument("--batch",   type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",      type=float, default=3e-4,
                        help="Learning rate warmup (default: 3e-4, finetune usa lr/10)")
    args = parser.parse_args()
    main(args)
