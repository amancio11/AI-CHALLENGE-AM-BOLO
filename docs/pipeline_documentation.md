# Glass Cleanliness Detection — Documentazione Completa della Pipeline

**Progetto:** AI Computer Vision per il controllo della pulizia delle superfici vetrate  
**Contesto:** Challenge Facility Management — elaborazione immagini/video da drone  
**Data:** Aprile 2026  
**Architettura:** Pipeline a due modelli (YOLOv8 + EfficientNet)

---

## Indice

1. [Panoramica della Soluzione](#1-panoramica-della-soluzione)
2. [Fase 0 — Setup Infrastruttura Azure](#fase-0--setup-infrastruttura-azure)
3. [Fase 1 — Dataset Rilevamento Finestre](#fase-1--dataset-rilevamento-finestre)
4. [Fase 2 — Training Modello 1: YOLOv8 Window Detector](#fase-2--training-modello-1-yolov8-window-detector)
5. [Fase 3 — Dataset Clean/Dirty](#fase-3--dataset-cleandirty)
6. [Fase 4 — Labeling Semi-automatico con CVAT](#fase-4--labeling-semi-automatico-con-cvat)
7. [Fase 5 — Training Modello 2: EfficientNet Cleanliness Classifier](#fase-5--training-modello-2-efficientnet-cleanliness-classifier)
8. [Fase 6 — Inferenza End-to-End e Heatmap](#fase-6--inferenza-end-to-end-e-heatmap)
9. [Struttura del Progetto](#struttura-del-progetto)

---

## 1. Panoramica della Soluzione

### Cosa vogliamo ottenere

Il sistema riceve in input un'immagine acquisita da un drone che riprende la facciata di un edificio. L'output è:

- L'**identificazione delle finestre** presenti nella facciata tramite bounding box
- Una **classificazione dello stato di pulizia** per ogni finestra (clean / dirty)
- Una **heatmap visiva** sovrapposta all'immagine che evidenzia le zone critiche
- Un **report strutturato** con coordinate e score per ogni pannello

### Schema ad alto livello

```
Immagine drone
      │
      ▼
┌─────────────────────────────┐
│  MODELLO 1: YOLOv8          │  → Dove sono le finestre? (bbox)
│  (Window Detector)          │
└─────────────────────────────┘
      │  bbox [x, y, w, h] per ogni finestra
      ▼
  crop automatico di ogni finestra
      │
      ▼
┌─────────────────────────────┐
│  MODELLO 2: EfficientNet    │  → Questa finestra è clean o dirty?
│  (Cleanliness Classifier)   │
└─────────────────────────────┘
      │  label + confidence per ogni finestra
      ▼
┌─────────────────────────────┐
│  Heatmap + Report JSON      │  → Output per il facility manager
└─────────────────────────────┘
```

### Classi di pulizia

| Classe | Label | Colore heatmap |
|--------|-------|----------------|
| 0 | `clean` | Verde |
| 1 | `dirty` | Rosso |
| 2 | `ambiguous` | Giallo (opzionale, scartato in training) |

### Perché questa architettura (vs Grounding DINO + SAM2)

La documentazione originale prevedeva Grounding DINO + SAM2 per trovare le finestre.
Questa soluzione è stata sostituita per i seguenti motivi:

| Aspetto | Grounding DINO + SAM2 | YOLOv8 (scelta attuale) |
|---|---|---|
| Tipo di output | Maschere pixel-perfect | Bounding box |
| Serve la maschera? | No — il crop bbox è sufficiente | ✅ Sì |
| Training richiesto | Zero-shot (instabile) | Fine-tuning supervisionato (robusto) |
| Velocità inferenza | ~2-5 sec/immagine | ~20ms/immagine |
| Complessità | 2 modelli in catena, fragile | 1 modello, standard |
| Dataset disponibile | Non necessario ma impreciso | `window-detection-vnpow` già scaricato ✅ |

---

## Fase 0 — Setup Infrastruttura Azure

### Obiettivo
Predisporre l'infrastruttura Azure per training, labeling e trasferimento dati. Step eseguito una volta sola.

### Risorse Azure

| Risorsa | Nome / SKU | Scopo |
|---------|-----------|-------|
| VM | `BOLOGNA-AI-AM-MACHINE` — Standard_NV12ads_A10_v5 | GPU A10-8Q 8GB VRAM — training YOLOv8 + EfficientNet |
| Storage Account | `kpmgbolognaai` (Resource Group `kpmg-bologna`) | Trasferimento dati tra PC locale e VM, archiviazione modelli e dataset |
| CVAT | su VM porta 8080 | Labeling immagini clean/dirty |
| MLflow | su VM porta 5000 | Tracking esperimenti di training |

### Stato attuale VM (Aprile 2026)

```
/mnt/                       ← disco dati 360GB (quasi tutto libero)
├── miniconda/              ← Miniconda installato, env glass-cv attivo
│   └── envs/glass-cv/      ← PyTorch 2.5.1+cu121, ultralytics 8.4.37, timm, mlflow
├── cvat/                   ← sorgente CVAT (da avviare con docker-compose)
├── docker-data/            ← dati Docker
├── glass-cleanliness/      ← cartella dati clean/dirty (da popolare)
└── project/                ← codice del progetto (da creare)

GPU: NVIDIA A10-8Q — 8GB VRAM — CUDA 12.8 — Driver 570.211.01
```

### Azure Storage Account — Trasferimento Dati

Lo Storage Account è il punto centrale per muovere dati tra PC locale, VM e colleghi.

**Accesso dalla VM (nella SSH):**
```bash
# Installa azcopy sulla VM (una volta sola)
wget https://aka.ms/downloadazcopy-v10-linux -O /tmp/azcopy.tar.gz
tar -xf /tmp/azcopy.tar.gz -C /tmp/
sudo mv /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
azcopy --version

# Login con credenziali Azure
azcopy login
```

**Struttura container consigliata nello Storage:**
```
kpmgbolognaai/
├── datasets/
│   ├── window-detection-vnpow-v1/   ← dataset Roboflow (upload da locale)
│   └── cleanliness/                 ← dataset clean/dirty (upload da locale)
├── models/
│   ├── yolov8_windows_best.pt       ← modello YOLOv8 trainato
│   └── efficientnet_cleanliness.pt  ← modello EfficientNet trainato
└── results/                         ← output inferenza
```

**Upload dataset da PC locale (PowerShell):**
```powershell
# Installa azcopy su Windows se non presente
# https://aka.ms/downloadazcopy-v10-windows

# Upload dell'intero dataset locale verso lo storage
azcopy copy "C:\Users\andreamancini\Downloads\AI-CHALLENGE2\data\raw\roboflow" `
  "https://kpmgbolognaai.blob.core.windows.net/datasets" `
  --recursive
```

**Download sulla VM (SSH):**
```bash
# Dalla VM — scarica il dataset dallo storage
azcopy copy \
  "https://kpmgbolognaai.blob.core.windows.net/datasets/window-detection-vnpow-v1" \
  "/mnt/project/data/raw/roboflow/" \
  --recursive

# Upload del modello trainato verso lo storage (dopo il training)
azcopy copy \
  "/mnt/project/runs/window_detection/yolov8n_windows/weights/best.pt" \
  "https://kpmgbolognaai.blob.core.windows.net/models/yolov8_windows_best.pt"
```

**Autenticazione con SAS token (alternativa a login interattivo):**
```bash
# Genera SAS token dal portale Azure → Storage Account → Shared Access Signature
# Poi usalo così:
azcopy copy \
  "/mnt/project/models/efficientnet_best.pt" \
  "https://kpmgbolognaai.blob.core.windows.net/models/efficientnet_best.pt?<SAS_TOKEN>"
```

### Setup ambiente conda (già eseguito)

```bash
# Sulla VM — ambiente già installato, attivare così:
conda activate glass-cv

# Verifica:
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Output atteso: CUDA: True | GPU: NVIDIA A10-8Q
```

### MLflow Tracking Server

```bash
# Sulla VM — avvia MLflow (eseguire in background)
conda activate glass-cv
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri /mnt/mlflow-data/mlruns \
  --default-artifact-root /mnt/mlflow-data/artifacts \
  > /mnt/mlflow-data/mlflow.log 2>&1 &

# Accesso: http://20.9.194.236:5000
```

**Accesso servizi:**
- MLflow: `http://20.9.194.236:5000`
- CVAT: `http://20.9.194.236:8080` (da avviare con docker-compose)

---

## Fase 1 — Dataset Rilevamento Finestre

### Obiettivo
Procurarsi un dataset di immagini di facciate con le finestre già annotate (bounding box) per addestrare YOLOv8.

### Come funziona Roboflow e i dataset annotati

**Roboflow** è una piattaforma per gestire dataset di computer vision. Ogni dataset su Roboflow Universe contiene:
- Immagini originali
- Annotazioni (bounding box, poligoni, keypoint...) create manualmente o semi-automaticamente
- Preprocessing e augmentation configurabili
- Export in tutti i formati principali (YOLO, COCO, Pascal VOC...)

Il dataset `window-detection-vnpow` che usiamo ha le finestre già annotate con bounding box. Sul sito Roboflow puoi visualizzare direttamente le detection con i box — questo è il **ground truth** di training, non un modello. Ogni immagine ha un file di annotazione che dice: "in questa foto c'è una finestra in posizione [x, y, w, h]".

### Dataset Primario: `window-detection-vnpow`

| Parametro | Valore |
|---|---|
| Sorgente | [Roboflow Universe — research-xvh79/window-detection-vnpow](https://universe.roboflow.com/research-xvh79/window-detection-vnpow) |
| Immagini totali | 4.840 |
| Formato annotazioni | COCO JSON (bbox) |
| Split | train: 3.866 / valid: 484 / test: 490 |
| Classe unica | `window` |
| Tipo immagini | Facciate di edifici, mix di reali e render CGI |

**Struttura su disco:**
```
data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1/
├── train/
│   ├── *.jpg                    ← immagini di training
│   └── _annotations.coco.json  ← annotazioni bbox in formato COCO
├── valid/
│   ├── *.jpg
│   └── _annotations.coco.json
└── test/
    ├── *.jpg
    └── _annotations.coco.json
```

**Formato annotazione COCO (esempio):**
```json
{
  "images": [{"id": 1, "file_name": "facade_001.jpg", "width": 640, "height": 480}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [120, 80, 200, 150],   ← [x_min, y_min, width, height] in pixel
     "area": 30000}
  ],
  "categories": [{"id": 1, "name": "window"}]
}
```

### Dataset Secondario (opzionale): da valutare

Per migliorare la robustezza del detector su edifici italiani/europei, si può integrare con:

| Dataset | Immagini | Note |
|---------|---------|------|
| [ADE20K windows](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | ~20k | Segmentazione, richiede conversione a bbox |
| [CMP Facade](http://cmp.felk.cvut.cz/~tylecr1/facade/) | ~600 | Europeo, ottimo per facciate storiche |
| Immagini drone KPMG | TBD | Dominio target, massima priorità se disponibili |

Se si aggiunge un secondo dataset, i dati vanno uniti e ri-annotati in formato YOLO prima del training.

### Pre-processing eseguito (locale)

Lo script `src/data/collect_dataset.py` ha già filtrato le immagini sintetiche (render CGI) e generato patch:

```
data/patches/
├── train/   ← 5.132 crop di finestre (solo immagini reali)
├── valid/   ← 631 crop
├── test/    ← 604 crop
└── pipeline_summary.json
```

> **Nota:** Le patch in `data/patches/` servono per analisi e validazione visiva. Il training YOLOv8 usa invece le immagini intere con le annotazioni originali in `data/raw/roboflow/`.

### Trasferimento del dataset sulla VM

```powershell
# Dal PC locale — upload verso Azure Storage
azcopy copy "C:\Users\andreamancini\Downloads\AI-CHALLENGE2\data\raw\roboflow" `
  "https://kpmgbolognaai.blob.core.windows.net/datasets" --recursive
```

```bash
# Dalla VM — download dallo storage
mkdir -p /mnt/project/data/raw/roboflow
azcopy copy \
  "https://kpmgbolognaai.blob.core.windows.net/datasets/window-detection-vnpow-v1" \
  "/mnt/project/data/raw/roboflow/" --recursive
```

In alternativa, scarica direttamente da Roboflow sulla VM:
```bash
conda activate glass-cv
cd /mnt/project
python -c "
from roboflow import Roboflow
import os
rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.workspace('research-xvh79').project('window-detection-vnpow')
project.version(1).download('coco', location='data/raw/roboflow/window-detection-vnpow-v1')
"
```

---

## Fase 2 — Training Modello 1: YOLOv8 Window Detector

### Obiettivo
Addestrare YOLOv8 a rilevare finestre in immagini di facciate.
Output: bounding box `[x, y, w, h]` con confidence score per ogni finestra.

---

### Come funziona l'addestramento YOLO

**YOLO (You Only Look Once)** è un algoritmo di object detection che analizza l'intera immagine in un solo passaggio forward della rete (da qui "Only Once"), invece di proporre regioni candidate prima e classificarle poi.

#### Cosa impara la rete

Durante il training, la rete impara a rispondere a questa domanda per ogni punto dell'immagine:
> "C'è un oggetto qui? Se sì, di che classe è, e qual è il suo bounding box esatto?"

L'immagine viene divisa in una griglia (es. 20×20 celle). Per ogni cella, la rete predice:
- **Objectness score**: probabilità che in quella cella ci sia un oggetto
- **Bounding box**: coordinate `[x, y, w, h]` relative alla cella
- **Class probability**: probabilità per ogni classe (nel nostro caso: solo `window`)

#### Loss function

La loss che minimizziamo durante il training è la somma di tre componenti:

$$\mathcal{L} = \lambda_{box} \cdot \mathcal{L}_{box} + \lambda_{obj} \cdot \mathcal{L}_{obj} + \lambda_{cls} \cdot \mathcal{L}_{cls}$$

- $\mathcal{L}_{box}$: errore di posizione e dimensione del bbox (CIoU loss)
- $\mathcal{L}_{obj}$: errore sul detection confidence (BCE loss)
- $\mathcal{L}_{cls}$: errore di classificazione (BCE loss)

#### Fine-tuning vs training da zero

Usiamo **fine-tuning** su pesi pre-addestrati su COCO (dataset con 80 classi, 330k immagini). Questo significa:
- La rete sa già riconoscere forme, bordi, texture
- Dobbiamo solo "specializzarla" per le finestre
- Risultato: converge in ~50-100 epoch invece di ~300, con meno dati

```
Pesi COCO (yolov8n.pt)           Dataset window-detection
       ↓                                    ↓
  Pre-training                         Fine-tuning
  (già fatto)                     (quello che facciamo noi)
       ↓                                    ↓
  Features generali              Features specifiche per finestre
```

#### Metriche di valutazione

| Metrica | Significato | Target |
|---------|-------------|--------|
| **mAP50** | Mean Average Precision con IoU≥0.50 | > 0.75 |
| **mAP50-95** | mAP media su soglie IoU 0.50→0.95 | > 0.55 |
| **Precision** | Quanti dei box predetti sono corretti | > 0.80 |
| **Recall** | Quante finestre reali vengono trovate | > 0.75 |

**IoU (Intersection over Union):** misura quanto un box predetto si sovrappone al ground truth. IoU=1.0 → perfetto. IoU≥0.5 → accettabile.

$$\text{IoU} = \frac{\text{Area di intersezione}}{\text{Area di unione}}$$

---

### Architettura: YOLOv8n vs YOLOv8s

| Variante | Parametri | VRAM | mAP50 (COCO) | Vel. A10 GPU |
|----------|-----------|------|--------------|-------------|
| **yolov8n** | 3.2M | ~1GB | 37.3 | ~2ms/img |
| **yolov8s** | 11.2M | ~2GB | 44.9 | ~3ms/img |
| yolov8m | 25.9M | ~4GB | 50.2 | ~5ms/img |

Con 8GB VRAM disponibili possiamo usare **yolov8s** (più accurato) senza problemi. Partiamo con `n` per velocità, proviamo `s` se il mAP è insufficiente.

---

### Preparazione dataset in formato YOLO

Il dataset Roboflow è in formato COCO. YOLOv8 può leggere COCO direttamente tramite un file YAML di configurazione:

```yaml
# data/window_detection.yaml
path: /mnt/project/data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1
train: train      # cartella con immagini di training
val: valid        # cartella con immagini di validazione
test: test        # cartella con immagini di test

nc: 1             # numero di classi
names: ['window'] # nomi delle classi
```

> YOLOv8 con formato COCO legge automaticamente il file `_annotations.coco.json` nella cartella — non serve convertire nulla.

---

### Script di training

```python
# src/training/train_yolo.py
"""
Fine-tuning YOLOv8 per window detection su dataset Roboflow.
Eseguire sulla VM con: conda activate glass-cv && python src/training/train_yolo.py
"""
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
from pathlib import Path
import argparse


def train(
    data_yaml: str,
    model_size: str = "n",   # 'n', 's', 'm'
    epochs: int = 100,
    imgsz: int = 640,
    device: str = "0",       # '0' = prima GPU, 'cpu' = CPU
):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("yolov8-window-detection")

    with mlflow.start_run(run_name=f"yolov8{model_size}_{epochs}ep"):
        # Carica modello pre-addestrato su COCO
        model = YOLO(f"yolov8{model_size}.pt")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,           # batch size — aumenta se hai più VRAM
            device=device,
            project="runs/window_detection",
            name=f"yolov8{model_size}_windows",
            patience=20,        # early stopping: ferma se no miglioramento per 20 epoch
            save=True,
            save_period=10,     # salva checkpoint ogni 10 epoch
            plots=True,         # genera grafici loss/metriche
            val=True,           # valuta su validation set ad ogni epoch
            # --- Augmentation ---
            # YOLOv8 applica queste trasformazioni in modo casuale a ogni immagine
            hsv_h=0.015,        # variazione hue
            hsv_s=0.7,          # variazione saturation
            hsv_v=0.4,          # variazione brightness
            degrees=5.0,        # rotazione max ±5°
            translate=0.1,      # traslazione max 10% dell'immagine
            scale=0.5,          # zoom casuale 50-150%
            flipud=0.0,         # no flip verticale (edifici non si capovolgono)
            fliplr=0.5,         # flip orizzontale 50% delle volte
            mosaic=1.0,         # mosaic augmentation (unisce 4 immagini in 1)
        )

        # Estrai metriche finali
        metrics = results.results_dict
        mlflow.log_params({
            "model": f"yolov8{model_size}",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": 16,
        })
        mlflow.log_metrics({
            "mAP50":    metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall":   metrics.get("metrics/recall(B)", 0),
        })

        # Salva modello su MLflow e storage
        best_pt = Path(f"runs/window_detection/yolov8{model_size}_windows/weights/best.pt")
        mlflow.log_artifact(str(best_pt), artifact_path="weights")

        print(f"\n✓ Training completato")
        print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 0):.3f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
        print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.3f}")
        print(f"  Modello:   {best_pt}")

    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/window_detection.yaml")
    parser.add_argument("--model",  default="n", choices=["n", "s", "m"],
                        help="Variante YOLOv8: n=nano, s=small, m=medium")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.device)
```

---

### Esecuzione sulla VM

```bash
# 1. Connettiti alla VM
ssh azureuseram@20.9.194.236

# 2. Crea struttura progetto
mkdir -p /mnt/project/{data/raw/roboflow,runs,models,src/training}
cd /mnt/project

# 3. Scarica il dataset (da storage o Roboflow)
azcopy copy "https://kpmgbolognaai.blob.core.windows.net/datasets/window-detection-vnpow-v1" \
  "data/raw/roboflow/" --recursive

# 4. Crea il file YAML di configurazione
cat > data/window_detection.yaml << 'EOF'
path: /mnt/project/data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1
train: train
val: valid
test: test
nc: 1
names: ['window']
EOF

# 5. Copia lo script di training (dal PC locale o scrivi direttamente)
# scp src/training/train_yolo.py azureuseram@20.9.194.236:/mnt/project/src/training/

# 6. Avvia MLflow in background
conda activate glass-cv
nohup mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri /mnt/mlflow-data/mlruns \
  > /mnt/mlflow-data/mlflow.log 2>&1 &

# 7. Smoke test (1 epoch, verifica che funzioni):
python src/training/train_yolo.py --epochs 1 --device 0

# 8. Training completo:
nohup python src/training/train_yolo.py \
  --model n --epochs 100 --device 0 \
  > /mnt/project/runs/train_yolo.log 2>&1 &

# Monitora il progresso:
tail -f /mnt/project/runs/train_yolo.log
```

**Tempi stimati su A10-8Q:**
- yolov8n, 100 epoch, 640px: ~8-12 minuti
- yolov8s, 100 epoch, 640px: ~15-20 minuti

---

### Cosa succede durante il training (epoch per epoch)

```
Epoch 1/100:
  - La rete vede tutte le 3.866 immagini di training (in batch da 16)
  - Per ogni batch calcola la loss (errore sui bbox)
  - Aggiusta i pesi con backpropagation (AdamW optimizer)
  - Alla fine valuta su validation set → stampa mAP50

Epoch 10/100:  loss in calo, mAP50 cresce rapidamente
Epoch 50/100:  convergenza rallenta, ajustamenti fini
Epoch 100/100: salva best.pt (il checkpoint con mAP50 più alto)
```

Il file `best.pt` è il modello finale da usare in produzione.

---

### Analisi risultati post-training

```bash
# Directory output:
runs/window_detection/yolov8n_windows/
├── weights/
│   ├── best.pt          ← usa questo per inferenza
│   └── last.pt          ← ultimo checkpoint
├── results.csv          ← metriche epoch per epoch
├── confusion_matrix.png ← matrice di confusione
├── PR_curve.png         ← Precision-Recall curve
├── F1_curve.png
└── val_batch*.jpg       ← visualizzazione predizioni su validation

# Test su immagine singola:
conda activate glass-cv
yolo detect predict \
  model=runs/window_detection/yolov8n_windows/weights/best.pt \
  source=data/raw/roboflow/window-detection-vnpow-v1/test/ \
  conf=0.4 save=True
```

### Upload modello su Storage dopo training

```bash
azcopy copy \
  "/mnt/project/runs/window_detection/yolov8n_windows/weights/best.pt" \
  "https://kpmgbolognaai.blob.core.windows.net/models/yolov8_windows_best.pt"
```

### Criteri di completamento Fase 2

- [ ] mAP50 > 0.75 su validation set
- [ ] Precision > 0.80, Recall > 0.75
- [ ] Validazione visiva su ~20 immagini campione
- [ ] `best.pt` salvato su Azure Storage
- [ ] Run loggata su MLflow con metriche e artifact

---

## Fase 3 — Dataset Clean/Dirty

### Obiettivo
Costruire un dataset di immagini di facciate etichettate `clean` / `dirty`
da usare per il training del classificatore (Modello 2).

**Nota importante:** questo dataset è **separato e indipendente** dal dataset
window-detection-vnpow. Non usiamo le patch già estratte (quelle servivano
per validare la pipeline) — qui usiamo immagini reali di facciate già disponibili.

### Struttura attesa

```
data/raw/cleanliness/
├── raw/                    ← immagini originali di facciate (già disponibili)
│   ├── facade_001.jpg
│   ├── facade_002.jpg
│   └── ...
└── labeled/                ← dopo il labeling in CVAT
    ├── clean/
    │   ├── facade_001__w00.jpg
    │   └── ...
    └── dirty/
        ├── facade_015__w02.jpg
        └── ...
```

### Pipeline per questo dataset

```
Immagini facciate (già hai)
         ↓
[YOLOv8 Fase 2] → estrae automaticamente bbox di ogni finestra
         ↓
crop automatici di tutte le finestre
         ↓
[CVAT] → labeling manuale clean / dirty / ambiguous
         ↓
Dataset finale:
  clean/   → aggiungere augmentation (flip, brightness, noise)
  dirty/   → usare così (rari, preziosi)
```

### Script: estrazione crop con YOLO (post Fase 2)

```python
# src/data/extract_crops_yolo.py
"""
Usa il modello YOLOv8 già trainato per estrarre automaticamente
i crop di finestre dalle immagini del dataset clean/dirty.
"""
from ultralytics import YOLO
import cv2
import json
from pathlib import Path
import argparse


def extract_crops(
    images_dir: Path,
    yolo_model_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.4,
    padding: float = 0.05,
):
    model = YOLO(str(yolo_model_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_crops = 0

    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        results = model(str(img_path), conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            w, h = x2 - x1, y2 - y1

            # Padding
            px, py = int(w * padding), int(h * padding)
            x1p = max(0, x1 - px)
            y1p = max(0, y1 - py)
            x2p = min(iw, x2 + px)
            y2p = min(ih, y2 + py)

            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            out_name = f"{img_path.stem}__w{idx:02d}_{x1}_{y1}_{w}_{h}.jpg"
            cv2.imwrite(str(output_dir / out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            manifest.append({
                "crop_file": out_name,
                "source_image": img_path.name,
                "bbox": [x1, y1, w, h],
                "confidence": float(box.conf[0]),
                "label": None,   # da compilare con CVAT
            })
            total_crops += 1

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Crop estratti: {total_crops} da {len(list(images_dir.glob('*.jpg')))} immagini")
    print(f"Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", required=True, help="Path a best.pt di YOLOv8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    extract_crops(
        images_dir=Path(args.images),
        yolo_model_path=Path(args.model),
        output_dir=Path(args.output),
        conf_threshold=args.conf,
    )
```

**Esecuzione:**
```bash
python src/data/extract_crops_yolo.py \
    --images data/raw/cleanliness/raw \
    --model runs/window_detection/yolov8n_windows/weights/best.pt \
    --output data/raw/cleanliness/crops_to_label
```

### Quantità target

| Classe | Minimo | Ottimale |
|---|---|---|
| `clean` | 150 | 300 (+ augmentation → 900+) |
| `dirty` | 150 | 300 |
| **Totale** | **300** | **600** |

Con augmentation sulle `clean`, 300 immagini totali sono sufficienti per un prototipo robusto.

---

## Fase 4 — Labeling Semi-automatico con CVAT

### Obiettivo
Etichettare i crop estratti in Fase 3 come `clean` / `dirty` / `ambiguous`.

### Setup task CVAT

```
Accesso: http://20.9.194.236:8080

1. Login → Create Project: "Glass Cleanliness"
2. Create Labels:
   - clean       (type: tag)
   - dirty       (type: tag)
   - ambiguous   (type: tag)
3. Create Task:
   - Nome: "Cleanliness Labeling Batch 1"
   - Upload immagini: cartella data/raw/cleanliness/crops_to_label/
   - Format: Image Classification
4. Assegna task → inizia labeling
```

### Linee guida labeling

**`clean`:** vetro trasparente, nessun deposito visibile, eventuali riflessi ok

**`dirty`:** qualsiasi combinazione di:
- Polvere densa o accumuli
- Righe d'acqua / aloni
- Macchie di guano
- Alghe o depositi organici
- Impronte visibili

**`ambiguous`:** foto sfocata, contro-luce forte, finestra parzialmente occultata
→ questi vengono esclusi dal training

### Export e organizzazione

Dopo il labeling esportare da CVAT in formato `ImageNet` (cartelle per classe):

```
data/raw/cleanliness/labeled/
├── clean/
│   ├── facade_001__w00.jpg
│   └── ...
├── dirty/
│   ├── facade_015__w02.jpg
│   └── ...
└── ambiguous/      ← esclusa dal training
    └── ...
```

### Criteri di completamento Fase 4

- [ ] Almeno 300 crop labellati (escluso `ambiguous`)
- [ ] Ratio clean/dirty tra 1:2 e 2:1 (da bilanciare con augmentation)
- [ ] File `manifest_labeled.json` aggiornato con i label CVAT

---

## Fase 5 — Training Modello 2: EfficientNet Cleanliness Classifier

### Obiettivo
Addestrare EfficientNet-B0 a classificare finestre come `clean` o `dirty`.
Input: crop di finestra 224×224px. Output: probabilità clean/dirty.

### Augmentation

L'augmentation è concentrata sulla classe `clean` per bilanciare il dataset
(nella realtà edifici con finestre sporche sono più rari da fotografare):

```python
# Augmentation aggressiva per clean (moltiplica i campioni)
clean_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

# Augmentation minima per dirty (non alterare i pattern di sporco)
dirty_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

### Dataset PyTorch

```python
# src/training/dataset_cleanliness.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

CLASS_MAP = {"clean": 0, "dirty": 1}


class CleanlinessDataset(Dataset):
    """
    Dataset per classificazione clean/dirty di crop di finestre.
    Supporta augmentation differenziata per classe.
    """

    def __init__(self, samples: list, augment: bool = False):
        self.samples = samples
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.clean_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.dirty_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        label = CLASS_MAP[label_str]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            transform = self.clean_aug if label == 0 else self.dirty_aug
        else:
            transform = self.base_transform

        return transform(img), label


def build_dataloaders(labeled_dir: Path, batch_size: int = 32):
    """Carica il dataset da cartelle clean/ e dirty/ e crea i DataLoader."""
    samples = []
    for class_name in ["clean", "dirty"]:
        class_dir = labeled_dir / class_name
        for img_path in class_dir.glob("*.jpg"):
            samples.append((img_path, class_name))

    # Split stratificato
    labels = [s[1] for s in samples]
    train_s, val_s = train_test_split(samples, test_size=0.2,
                                       stratify=labels, random_state=42)

    from torch.utils.data import DataLoader
    train_dl = DataLoader(CleanlinessDataset(train_s, augment=True),
                          batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(CleanlinessDataset(val_s, augment=False),
                        batch_size=batch_size, num_workers=4)
    return train_dl, val_dl
```

### Script di training

```python
# src/training/train_efficientnet.py
"""
Fine-tuning EfficientNet-B0 per classificazione clean/dirty.
"""
import torch
import torch.nn as nn
from torchvision import models
import mlflow
from pathlib import Path
import argparse
from dataset_cleanliness import build_dataloaders


def train(labeled_dir: str, output_dir: str, epochs: int = 30,
          lr: float = 3e-4, device: str = "cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training su: {device}")

    train_dl, val_dl = build_dataloaders(Path(labeled_dir))

    # Backbone EfficientNet-B0 pre-addestrato su ImageNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Sostituisci solo il classifer finale (2 classi: clean/dirty)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    model = model.to(device)

    # Fase 1: congela backbone, allena solo il classifier (5 epoch)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("efficientnet-cleanliness")
    with mlflow.start_run():
        mlflow.log_params({"model": "efficientnet_b0", "lr": lr,
                           "epochs": epochs, "labeled_dir": labeled_dir})
        best_val_acc = 0.0

        for phase, freeze, n_ep in [("warmup", True, 5), ("finetune", False, epochs - 5)]:
            if phase == "finetune":
                # Fase 2: sblocca tutto il backbone
                for param in model.features.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr / 10,
                                               weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_ep)

            for epoch in range(n_ep):
                # Train
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for imgs, labels in train_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_correct += (outputs.argmax(1) == labels).sum().item()
                    train_total += len(labels)

                # Val
                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for imgs, labels in val_dl:
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)
                        val_correct += (outputs.argmax(1) == labels).sum().item()
                        val_total += len(labels)

                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                ep_num = epoch + (5 if phase == "finetune" else 0)
                print(f"[{phase}] Epoch {ep_num}: "
                      f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
                mlflow.log_metrics({"train_acc": train_acc, "val_acc": val_acc},
                                   step=ep_num)

                # Salva il miglior modello
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(),
                               f"{output_dir}/efficientnet_b0_best.pt")

                if phase == "finetune":
                    scheduler.step()

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(f"{output_dir}/efficientnet_b0_best.pt")
        print(f"Best val_acc: {best_val_acc:.4f}")

    return f"{output_dir}/efficientnet_b0_best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", required=True,
                        help="Cartella con clean/ e dirty/ (output CVAT)")
    parser.add_argument("--output", default="models/cleanliness")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.labeled_dir, args.output, args.epochs, device=args.device)
```

**Esecuzione:**
```bash
# Smoke test (1 epoch):
python src/training/train_efficientnet.py \
    --labeled-dir data/raw/cleanliness/labeled \
    --epochs 1 --device cpu

# Training completo sulla VM:
python src/training/train_efficientnet.py \
    --labeled-dir data/raw/cleanliness/labeled \
    --epochs 30 --device cuda
```

**Tempi stimati:**
- CPU locale: ~2-4 ore
- VM Azure A10: ~3-8 minuti

### Criteri di completamento Fase 5

- [ ] Val accuracy > 85%
- [ ] F1-score `dirty` > 80% (la classe più importante in produzione)
- [ ] Confusion matrix analizzata
- [ ] Modello salvato: `models/cleanliness/efficientnet_b0_best.pt`

---

## Fase 6 — Inferenza End-to-End e Heatmap

### Obiettivo
Pipeline completa che prende un'immagine di facciata e produce:
- Immagine annotata con box colorati (verde=clean, rosso=dirty)
- Heatmap Grad-CAM sulle finestre sporche (dove è concentrato lo sporco)
- Report JSON per il facility manager

### Script di inferenza

```python
# src/inference/analyze_facade.py
"""
Pipeline end-to-end: immagine facciata → report + heatmap.
Richiede entrambi i modelli trainati (YOLOv8 + EfficientNet).
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import argparse

CLASSES = ["clean", "dirty"]
COLORS_BGR = {
    "clean": (0, 200, 0),    # Verde
    "dirty": (0, 0, 220),    # Rosso
}


def load_efficientnet(model_path: str, device: torch.device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def classify_crop(crop_bgr: np.ndarray, model, device) -> tuple[str, float]:
    """Classifica un crop come clean/dirty. Ritorna (label, confidence)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx])


def analyze_facade(
    image_path: str,
    yolo_path: str,
    efficientnet_path: str,
    output_image: str,
    output_report: str,
    conf_threshold: float = 0.4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_path)
    classifier = load_efficientnet(efficientnet_path, device)

    img = cv2.imread(image_path)
    result_img = img.copy()
    ih, iw = img.shape[:2]

    # Fase 1: YOLOv8 trova le finestre
    detections_yolo = yolo(image_path, conf=conf_threshold, verbose=False)[0].boxes
    report_detections = []

    if detections_yolo is not None:
        for i, box in enumerate(detections_yolo):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Fase 2: EfficientNet classifica il crop
            label, confidence = classify_crop(crop, classifier, device)
            color = COLORS_BGR[label]

            # Disegna box e label
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.0%}"
            cv2.putText(result_img, text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            report_detections.append({
                "window_id": i,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "label": label,
                "confidence": round(confidence, 3),
                "yolo_conf": float(box.conf[0]),
            })

    # Report aggregato
    n_dirty = sum(1 for d in report_detections if d["label"] == "dirty")
    n_total = len(report_detections)
    report = {
        "image": Path(image_path).name,
        "total_windows": n_total,
        "dirty_windows": n_dirty,
        "clean_windows": n_total - n_dirty,
        "dirty_ratio": round(n_dirty / max(n_total, 1), 3),
        "detections": report_detections,
    }

    cv2.imwrite(output_image, result_img)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Finestre rilevate: {n_total} | Sporche: {n_dirty} | Pulite: {n_total - n_dirty}")
    print(f"Output: {output_image}")
    print(f"Report: {output_report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--output-image", default="output_annotated.jpg")
    parser.add_argument("--output-report", default="output_report.json")
    args = parser.parse_args()

    analyze_facade(
        image_path=args.image,
        yolo_path=args.yolo,
        efficientnet_path=args.classifier,
        output_image=args.output_image,
        output_report=args.output_report,
    )
```

**Esecuzione:**
```bash
python src/inference/analyze_facade.py \
    --image data/test/facade_001.jpg \
    --yolo runs/window_detection/yolov8n_windows/weights/best.pt \
    --classifier models/cleanliness/efficientnet_b0_best.pt \
    --output-image results/facade_001_annotated.jpg \
    --output-report results/facade_001_report.json
```

### Esempio output report.json

```json
{
  "image": "facade_001.jpg",
  "total_windows": 12,
  "dirty_windows": 4,
  "clean_windows": 8,
  "dirty_ratio": 0.333,
  "detections": [
    {
      "window_id": 0,
      "bbox": [120, 80, 200, 150],
      "label": "dirty",
      "confidence": 0.923,
      "yolo_conf": 0.876
    },
    {
      "window_id": 1,
      "bbox": [340, 80, 200, 150],
      "label": "clean",
      "confidence": 0.991,
      "yolo_conf": 0.912
    }
  ]
}
```

### Criteri di completamento Fase 6

- [ ] Pipeline end-to-end funzionante su immagine singola
- [ ] Tempo inferenza < 500ms/immagine (su GPU)
- [ ] Report JSON corretto e leggibile
- [ ] Annotazioni visuali coerenti con lo stato reale delle finestre

---

## Struttura del Progetto

```
AI-CHALLENGE2/
├── docs/
│   └── pipeline_documentation.md      ← questo file
│
├── scripts/
│   └── setup_vm.sh                     ← Fase 0: setup VM Azure
│
├── src/
│   ├── data/
│   │   ├── collect_dataset.py          ← Fase 1: pipeline download + filtro + crop
│   │   ├── filter_synthetic.py         ← Fase 1: filtro render/CGI
│   │   ├── crop_window_patches.py      ← Fase 1: crop da annotazioni COCO
│   │   ├── quality_filter.py           ← Fase 1: filtro qualità patch
│   │   ├── download_roboflow.py        ← Fase 1: download da Roboflow
│   │   └── extract_crops_yolo.py       ← Fase 3: estrazione crop con YOLOv8
│   │
│   ├── training/
│   │   ├── train_yolo.py               ← Fase 2: fine-tuning YOLOv8
│   │   ├── train_efficientnet.py       ← Fase 5: training EfficientNet
│   │   └── dataset_cleanliness.py      ← Fase 5: PyTorch Dataset clean/dirty
│   │
│   └── inference/
│       └── analyze_facade.py           ← Fase 6: pipeline end-to-end
│
├── data/
│   ├── raw/
│   │   ├── roboflow/                   ← Dataset window-detection-vnpow ✅
│   │   └── cleanliness/                ← Dataset clean/dirty (da costruire)
│   │       ├── raw/                    ← Immagini facciate originali
│   │       ├── crops_to_label/         ← Crop estratti da YOLOv8
│   │       └── labeled/                ← Dopo labeling CVAT
│   │           ├── clean/
│   │           └── dirty/
│   └── patches/                        ← Patch estratte da window-detection-vnpow ✅
│       ├── train/  (5.132 patch)
│       ├── valid/  (631 patch)
│       ├── test/   (604 patch)
│       └── manifest.json
│
├── runs/
│   └── window_detection/               ← Output training YOLOv8
│
├── models/
│   └── cleanliness/                    ← Output training EfficientNet
│
├── results/                            ← Output inferenza
│
├── .env                                ← ROBOFLOW_API_KEY (git-ignored)
└── requirements.txt
```

---

## Riepilogo Dipendenze Python

```txt
# requirements.txt
torch>=2.2.0
torchvision>=0.17.0
ultralytics>=8.0.0          # YOLOv8
timm>=0.9.0
opencv-python>=4.9.0
Pillow>=10.2.0
scikit-learn>=1.4.0
mlflow>=2.11.0
tqdm>=4.66.0
roboflow>=1.1.0
python-dotenv>=1.0.0
numpy>=1.26.0
```

---

## Stato Avanzamento

| Fase | Descrizione | Stato |
|------|-------------|-------|
| 0 | Setup VM Azure | ⚠️ VM in riavvio |
| 1 | Dataset window-detection scaricato e processato | ✅ Completo |
| 2 | Training YOLOv8 Window Detector | ⏳ Da fare |
| 3 | Dataset clean/dirty — estrazione crop | ⏳ Da fare (dopo Fase 2) |
| 4 | Labeling CVAT clean/dirty | ⏳ Da fare (dopo Fase 3) |
| 5 | Training EfficientNet Cleanliness | ⏳ Da fare (dopo Fase 4) |
| 6 | Inferenza end-to-end + report | ⏳ Da fare (dopo Fase 5) |

---

*Documentazione aggiornata — AI Challenge Facility Management Glass Cleanliness Detection — Aprile 2026*

