# Fase 1 — Training YOLOv8 Window Detector
## Documentazione Tecnica e Funzionale

**Progetto:** Glass Cleanliness Detection — AI Computer Vision  
**Step:** 1 di 6 — Rilevamento automatico delle finestre nelle facciate  
**Data:** Aprile 2026  
**Prerequisito:** Fase 0 completata (VM online, conda glass-cv attivo, GPU disponibile)

---

## Indice

1. [Obiettivo](#1-obiettivo)
2. [Cosa è COCO e Cosa è YOLOv8 — Chiarimento Fondamentale](#2-cosa-è-coco-e-cosa-è-yolov8--chiarimento-fondamentale)
3. [Il Dataset: window-detection-vnpow](#3-il-dataset-window-detection-vnpow)
4. [Come Funziona l'Addestramento YOLOv8](#4-come-funziona-laddestramento-yolov8)
5. [Preparazione dell'Ambiente sulla VM](#5-preparazione-dellambiente-sulla-vm)
6. [Script di Training](#6-script-di-training)
7. [Esecuzione e Monitoraggio](#7-esecuzione-e-monitoraggio)
8. [Analisi dei Risultati](#8-analisi-dei-risultati)
9. [Dataset Aggiuntivi (opzionale)](#9-dataset-aggiuntivi-opzionale)
10. [Criteri di Completamento](#10-criteri-di-completamento)

---

## 1. Obiettivo

Addestrare un modello YOLOv8 capace di localizzare tutte le finestre presenti in un'immagine di facciata di edificio, producendo per ognuna un **bounding box** `[x, y, w, h]` con relativo **confidence score**.

Questo modello è il primo dei due stadi della pipeline:

```
Immagine facciata
      │
      ▼
 [YOLOv8]  ←── questo è ciò che addestriamo in questa fase
      │
      ▼
bbox di ogni finestra → crop → [EfficientNet] → clean / dirty
```

---

## 2. Cosa è COCO e Cosa è YOLOv8 — Chiarimento Fondamentale

Questo punto causa spesso confusione. È importante distinguere **tre cose separate**:

### 2.1 Il dataset COCO (Microsoft)

**COCO** (Common Objects in Context) è un dataset pubblico di Microsoft con:
- ~330.000 immagini di scene quotidiane
- 80 classi di oggetti: persone, auto, sedie, biciclette, animali...
- Annotazioni: bounding box + segmentazione

**Non lo usiamo direttamente.** Non scaricheremo il dataset COCO.

### 2.2 Il "formato COCO" per le annotazioni

Separatamente dal dataset, il **formato JSON di COCO** è diventato lo standard de facto per distribuire annotazioni. Contiene:

```json
{
  "images": [
    {"id": 1, "file_name": "facade_001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 80, 200, 150],
      "area": 30000
    }
  ],
  "categories": [
    {"id": 1, "name": "window"}
  ]
}
```

**Questo SÌ lo usiamo** — il dataset Roboflow che abbiamo scaricato usa questo formato per descrivere dove sono le finestre nelle immagini.

> Il "formato COCO" è solo una convenzione JSON. Non ha nulla a che fare con le 80 classi del dataset COCO originale.

### 2.3 I pesi pre-addestrati `yolov8n.pt`

Quando scriviamo `YOLO("yolov8n.pt")`, Ultralytics scarica automaticamente un file di pesi (parametri della rete) che sono stati ottimizzati sul dataset COCO originale (80 classi). Questi pesi:
- **Non contengono immagini del dataset COCO** — solo i parametri della rete
- Codificano la conoscenza generale di come riconoscere oggetti (bordi, forme, texture)
- Vengono usati come punto di partenza per il **fine-tuning** sul nostro dataset

```
yolov8n.pt  ←── scarica Ultralytics automaticamente (~6MB)
     │
     │  fine-tuning su window-detection-vnpow (4840 immagini, 1 classe)
     ▼
yolov8n_windows_best.pt  ←── il nostro modello finale
```

### Riepilogo visivo

| Cosa | Cosa è | Lo usiamo? |
|------|--------|-----------|
| Dataset COCO (Microsoft) | 330k immagini, 80 classi | ❌ No |
| Formato COCO (JSON) | Standard file annotazioni | ✅ Sì (il nostro dataset Roboflow) |
| Pesi `yolov8n.pt` | Parametri rete pre-addestrata | ✅ Sì (punto di partenza fine-tuning) |
| `window-detection-vnpow` | Il nostro dataset (finestre) | ✅ Sì (training supervisionato) |

---

## 3. Il Dataset: window-detection-vnpow

### Provenienza

Dataset pubblico disponibile su [Roboflow Universe](https://universe.roboflow.com/research-xvh79/window-detection-vnpow), creato dalla community per il rilevamento di finestre nelle facciate.

### Statistiche

| Split | Immagini totali | Immagini reali* | Finestre annotate (stima) |
|-------|----------------|-----------------|--------------------------|
| Train | 3.866 | ~2.254 (58%) | ~15.000 |
| Valid | 484 | ~284 (59%) | ~1.800 |
| Test | 490 | ~293 (60%) | ~1.900 |
| **Totale** | **4.840** | **~2.831** | **~18.700** |

*Immagini reali = filtrate da render CGI tramite analisi dell'entropia cromatica

### Struttura su disco

```
data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1/
├── train/
│   ├── facade_001.jpg
│   ├── facade_002.jpg
│   ├── ...
│   └── _annotations.coco.json     ← annotazioni bbox per TUTTE le immagini del split
├── valid/
│   ├── ...
│   └── _annotations.coco.json
└── test/
    ├── ...
    └── _annotations.coco.json
```

### Cosa contiene il file _annotations.coco.json

Il file è un JSON con tre array principali:

```json
{
  "info": {...},
  "licenses": [...],
  "categories": [{"id": 0, "name": "window", "supercategory": "none"}],
  "images": [
    {
      "id": 0,
      "license": 1,
      "file_name": "facade_001.jpg",
      "height": 640,
      "width": 640
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 0,
      "bbox": [86, 142, 98, 87],    ← [x_min, y_min, width, height] in pixel
      "area": 8526,
      "segmentation": [],
      "iscrowd": 0
    },
    ...
  ]
}
```

**Ogni annotazione** = una finestra in una immagine specifica, descritta dal suo bounding box rettangolare.

### Visualizzare il dataset

Sul sito Roboflow (o localmente con un piccolo script) si possono vedere le immagini con i box sovrapposti. Quello che vedi sul sito è il **ground truth** — le annotazioni manuali originali, non predizioni di un modello.

```python
# Script rapido per visualizzare le annotazioni localmente
import json, cv2, random
from pathlib import Path

annot_path = "data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1/train/_annotations.coco.json"
img_dir = "data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1/train"

with open(annot_path) as f:
    data = json.load(f)

# Scegli immagine casuale
img_info = random.choice(data["images"])
img = cv2.imread(f"{img_dir}/{img_info['file_name']}")

# Disegna tutti i bbox di quell'immagine
for ann in data["annotations"]:
    if ann["image_id"] == img_info["id"]:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("preview.jpg", img)
print(f"Immagine: {img_info['file_name']} → preview.jpg")
```

---

## 4. Come Funziona l'Addestramento YOLOv8

### 4.1 YOLO in una riga

> YOLO divide l'immagine in una griglia e per ogni cella predice: "c'è un oggetto qui? Di che classe? Dove esattamente?"

### 4.2 Architettura della rete

YOLOv8 è composta da tre parti:

```
Input (640×640 px)
      │
      ▼
┌──────────────────┐
│  BACKBONE        │  Estrae feature (bordi, texture, forme)
│  (CSPDarknet)    │  ← pre-addestrato su COCO, si aggiorna poco
└──────────────────┘
      │
      ▼
┌──────────────────┐
│  NECK            │  Combina feature a diverse scale
│  (PAN-FPN)       │  per rilevare finestre grandi e piccole
└──────────────────┘
      │
      ▼
┌──────────────────┐
│  HEAD            │  Predice: bbox + confidence + classe
│  (Detection)     │  ← si specializza completamente nel fine-tuning
└──────────────────┘
      │
      ▼
Output: lista di [x, y, w, h, confidence, classe] per ogni finestra
```

### 4.3 Loss function

Durante il training, la rete impara minimizzando questa loss:

$$\mathcal{L} = \lambda_{box} \cdot \mathcal{L}_{box} + \lambda_{obj} \cdot \mathcal{L}_{obj} + \lambda_{cls} \cdot \mathcal{L}_{cls}$$

| Componente | Cosa misura | Formula |
|---|---|---|
| $\mathcal{L}_{box}$ | Quanto il box predetto si discosta dal ground truth | CIoU loss |
| $\mathcal{L}_{obj}$ | Quanto è sicura la rete che ci sia un oggetto | Binary Cross-Entropy |
| $\mathcal{L}_{cls}$ | Quanto è corretta la classificazione della classe | Binary Cross-Entropy |

### 4.4 Fine-tuning: perché funziona con pochi dati

Il fine-tuning sfrutta la **transfer learning**: i layer early del backbone (che riconoscono bordi e texture generici) vengono "congelati" o aggiornati poco. Solo i layer finali vengono specializzati per "finestre".

```
Epoch 1-10:   backpropagation modifica principalmente il HEAD
Epoch 10-50:  modifica gradualmente anche il NECK
Epoch 50-100: fine-tuning globale con learning rate decrescente
```

Con il dataset window-detection-vnpow (4840 immagini) questo converge in ~50-100 epoch.

### 4.5 Augmentation (data augmentation)

YOLOv8 applica trasformazioni casuali a ogni immagine durante il training per rendere il modello più robusto. Configuriamo la augmentation in base al nostro dominio:

| Parametro | Valore | Motivazione |
|---|---|---|
| `fliplr=0.5` | Flip orizzontale 50% | Gli edifici sono simmetrici |
| `flipud=0.0` | No flip verticale | Le finestre hanno un "sopra" |
| `degrees=5.0` | Rotazione ±5° | Drone leggermente inclinato |
| `hsv_v=0.4` | Variazione luminosità | Ore del giorno diverse |
| `scale=0.5` | Zoom ±50% | Distanza variabile dal drone |
| `mosaic=1.0` | Unisce 4 immagini in 1 | Aumenta la varietà di contesti |

### 4.6 Metriche di valutazione

| Metrica | Formula | Target |
|---|---|---|
| **mAP50** | Area sotto la PR curve a IoU≥0.50 | > 0.75 |
| **mAP50-95** | Media mAP da IoU=0.50 a 0.95 (step 0.05) | > 0.55 |
| **Precision** | TP / (TP + FP) | > 0.80 |
| **Recall** | TP / (TP + FN) | > 0.75 |

**IoU** (Intersection over Union):
$$\text{IoU} = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}$$

Un box è "corretto" se IoU ≥ 0.50 con il ground truth.

---

## 5. Preparazione dell'Ambiente sulla VM

### 5.1 Struttura directory

```bash
ssh azureuseram@20.9.194.236

mkdir -p /mnt/project/{data/raw/roboflow,runs,models,src/training,src/data}
cd /mnt/project
```

### 5.2 Trasferimento dataset

**Opzione A — da Azure Storage (se già caricato):**
```bash
# Installa azcopy se non presente
wget https://aka.ms/downloadazcopy-v10-linux -O /tmp/azcopy.tar.gz
tar -xf /tmp/azcopy.tar.gz -C /tmp/
sudo mv /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
azcopy login

azcopy copy \
  "https://stglasscleanliness.blob.core.windows.net/datasets/window-detection-vnpow-v1" \
  "/mnt/project/data/raw/roboflow/" --recursive
```

**Opzione B — download diretto da Roboflow (più rapido):**
```bash
conda activate glass-cv
cd /mnt/project

python - << 'EOF'
from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("research-xvh79").project("window-detection-vnpow")
project.version(1).download("coco", location="data/raw/roboflow/window-detection-vnpow-v1")
EOF
```

### 5.3 File YAML di configurazione

```bash
cat > /mnt/project/data/window_detection.yaml << 'EOF'
path: /mnt/project/data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1
train: train
val: valid
test: test

nc: 1
names: ['window']
EOF
```

### 5.4 Verifica dataset

```bash
conda activate glass-cv
python - << 'EOF'
import json
from pathlib import Path

base = Path("data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1")
for split in ["train", "valid", "test"]:
    annot = base / split / "_annotations.coco.json"
    with open(annot) as f:
        data = json.load(f)
    n_img = len(data["images"])
    n_ann = len(data["annotations"])
    print(f"{split}: {n_img} immagini, {n_ann} annotazioni, avg={n_ann/n_img:.1f} finestre/img")
EOF
```

Output atteso:
```
train: 3866 immagini, ~15000 annotazioni, avg~3.9 finestre/img
valid: 484 immagini, ~1800 annotazioni, avg~3.7 finestre/img
test:  490 immagini, ~1900 annotazioni, avg~3.9 finestre/img
```

---

## 6. Script di Training

```python
# src/training/train_yolo.py
"""
Fine-tuning YOLOv8 per window detection sul dataset Roboflow window-detection-vnpow.

Uso:
    conda activate glass-cv
    python src/training/train_yolo.py --model n --epochs 100 --device 0

Argomenti:
    --data    : path al file YAML di configurazione dataset
    --model   : variante YOLOv8 (n=nano, s=small, m=medium)
    --epochs  : numero di epoche di training
    --imgsz   : dimensione immagini in input (default 640)
    --device  : '0' per GPU, 'cpu' per CPU
"""
from ultralytics import YOLO
import mlflow
from pathlib import Path
import argparse


def train(
    data_yaml: str = "data/window_detection.yaml",
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    device: str = "0",
) -> Path:

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("yolov8-window-detection")

    with mlflow.start_run(run_name=f"yolov8{model_size}_{epochs}ep_img{imgsz}"):

        # Carica il modello pre-addestrato su COCO
        # yolov8n.pt viene scaricato automaticamente da Ultralytics (~6MB)
        model = YOLO(f"yolov8{model_size}.pt")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            device=device,
            project="runs/window_detection",
            name=f"yolov8{model_size}_windows",
            patience=20,       # early stopping: ferma dopo 20 epoch senza miglioramento
            save=True,
            save_period=10,    # checkpoint ogni 10 epoch
            plots=True,        # genera curve loss, PR, F1
            val=True,          # valuta su validation set ad ogni epoch
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,        # edifici non si capovolgono
            fliplr=0.5,
            mosaic=1.0,
        )

        # Log parametri e metriche su MLflow
        metrics = results.results_dict
        mlflow.log_params({
            "model": f"yolov8{model_size}",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": 16,
            "dataset": "window-detection-vnpow-v1",
        })
        mlflow.log_metrics({
            "mAP50":     metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95":  metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall":    metrics.get("metrics/recall(B)", 0),
        })

        best_pt = Path(f"runs/window_detection/yolov8{model_size}_windows/weights/best.pt")
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")

        print(f"\n✓ Training completato")
        print(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 0):.3f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
        print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.3f}")
        print(f"  Modello:   {best_pt}")

    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 per window detection")
    parser.add_argument("--data",   default="data/window_detection.yaml")
    parser.add_argument("--model",  default="n", choices=["n", "s", "m"],
                        help="n=nano(3.2M param), s=small(11M), m=medium(26M)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="0", help="'0'=GPU, 'cpu'=CPU")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.device)
```

---

## 7. Esecuzione e Monitoraggio

### 7.1 Avvio MLflow (prerequisito)

```bash
ssh azureuseram@20.9.194.236
conda activate glass-cv
mkdir -p /mnt/mlflow-data/{mlruns,artifacts}

nohup mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri /mnt/mlflow-data/mlruns \
  --default-artifact-root /mnt/mlflow-data/artifacts \
  > /mnt/mlflow-data/mlflow.log 2>&1 &

# Verifica: http://20.9.194.236:5000
```

### 7.2 Smoke test (1 epoch, verifica che tutto funzioni)

```bash
cd /mnt/project
conda activate glass-cv

python src/training/train_yolo.py --model n --epochs 1 --device 0
```

Durata attesa: ~30-60 secondi. Se completa senza errori, procedi.

### 7.3 Training completo in background

```bash
nohup python src/training/train_yolo.py \
  --model n \
  --epochs 100 \
  --device 0 \
  > /mnt/project/runs/train_yolo_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "PID: $!"
```

### 7.4 Monitoraggio in tempo reale

```bash
# Segui il log
tail -f /mnt/project/runs/train_yolo_*.log

# Oppure monitora le metriche YOLOv8 direttamente
tail -f /mnt/project/runs/window_detection/yolov8n_windows/results.csv

# GPU utilization
watch -n 5 nvidia-smi
```

### 7.5 Tempi stimati su A10-8Q (8GB VRAM)

| Variante | Epoch | Imgsz | Tempo totale |
|----------|-------|-------|-------------|
| yolov8n | 100 | 640 | ~8-12 min |
| yolov8s | 100 | 640 | ~15-20 min |
| yolov8n | 100 | 1280 | ~25-35 min |

### 7.6 Confronto varianti (strategia consigliata)

```bash
# Run 1: yolov8n (veloce, baseline)
python src/training/train_yolo.py --model n --epochs 100

# Se mAP50 < 0.75 → prova yolov8s
python src/training/train_yolo.py --model s --epochs 100

# Confronta i due run su MLflow: http://20.9.194.236:5000
```

---

## 8. Analisi dei Risultati

### 8.1 File di output

```
runs/window_detection/yolov8n_windows/
├── weights/
│   ├── best.pt          ← modello con il mAP50 più alto (usa questo)
│   └── last.pt          ← ultimo checkpoint
├── results.csv          ← metriche epoch per epoch
├── confusion_matrix.png ← falsi positivi / falsi negativi
├── PR_curve.png         ← curva Precision-Recall
├── F1_curve.png         ← curva F1 score per soglia di confidence
├── labels.jpg           ← distribuzione dimensioni bbox nel dataset
└── val_batch0_pred.jpg  ← visualizzazione predizioni su validation
```

### 8.2 Test su immagini reali

```bash
# Inferenza su tutto il test set
conda activate glass-cv
yolo detect predict \
  model=runs/window_detection/yolov8n_windows/weights/best.pt \
  source=data/raw/roboflow/window-detection-vnpow/window-detection-vnpow-v1/test \
  conf=0.4 \
  save=True \
  project=runs/predictions \
  name=test_set

# Le immagini annotate sono in runs/predictions/test_set/
```

### 8.3 Test su immagine singola

```bash
yolo detect predict \
  model=runs/window_detection/yolov8n_windows/weights/best.pt \
  source=path/alla/tua/immagine.jpg \
  conf=0.3 \
  save=True
```

### 8.4 Script di validazione con metriche

```python
# Valutazione completa sul test set
from ultralytics import YOLO

model = YOLO("runs/window_detection/yolov8n_windows/weights/best.pt")
metrics = model.val(
    data="data/window_detection.yaml",
    split="test",
    conf=0.4,
    iou=0.5,
)
print(f"mAP50:    {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision:{metrics.box.mp:.3f}")
print(f"Recall:   {metrics.box.mr:.3f}")
```

### 8.5 Upload modello su Storage

```bash
# Dopo training completato con successo
azcopy copy \
  "/mnt/project/runs/window_detection/yolov8n_windows/weights/best.pt" \
  "https://stglasscleanliness.blob.core.windows.net/models/yolov8_windows_best.pt"

echo "✓ Modello caricato su Azure Storage"
```

---

## 9. Dataset Aggiuntivi (opzionale)

Se il mAP50 non raggiunge il target (>0.75), si può integrare il dataset con immagini aggiuntive di facciate europee/italiane — più simili al dominio target (edifici KPMG).

| Dataset | Immagini | Formato | Note |
|---------|---------|---------|------|
| [CMP Facade DB](http://cmp.felk.cvut.cz/~tylecr1/facade/) | 606 | PNG + labeling | Europeo, facciate storiche |
| [ECP Facade](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) | ~100 | Varie | Alta qualità |
| Immagini drone KPMG | TBD | Da annotare | Massima priorità: dominio target |

### Integrazione dataset aggiuntivo

1. Annotare le nuove immagini in CVAT esportando in formato COCO
2. Unire i JSON di annotazione con lo script:

```python
# Merge di due dataset COCO
import json

def merge_coco(datasets: list[str], output: str):
    merged = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "window"}]}
    img_id, ann_id = 0, 0

    for ds_path in datasets:
        with open(ds_path) as f:
            data = json.load(f)

        # Remap IDs per evitare conflitti
        id_map = {}
        for img in data["images"]:
            id_map[img["id"]] = img_id
            img["id"] = img_id
            merged["images"].append(img)
            img_id += 1

        for ann in data["annotations"]:
            ann["image_id"] = id_map[ann["image_id"]]
            ann["id"] = ann_id
            ann["category_id"] = 0
            merged["annotations"].append(ann)
            ann_id += 1

    with open(output, "w") as f:
        json.dump(merged, f)
    print(f"Merged: {len(merged['images'])} immagini, {len(merged['annotations'])} annotazioni")
```

---

## 10. Criteri di Completamento

La Fase 1 è completata quando:

- [ ] Dataset scaricato sulla VM: `data/raw/roboflow/window-detection-vnpow-v1/`
- [ ] Smoke test completato senza errori (1 epoch)
- [ ] Training completo: 100 epoch, convergenza osservata su MLflow
- [ ] **mAP50 > 0.75** sul validation set
- [ ] Precision > 0.80, Recall > 0.75
- [ ] Validazione visiva: predizioni corrette su almeno 20 immagini campione
- [ ] `best.pt` caricato su Azure Storage
- [ ] Run MLflow documentata con metriche e artifact

### Prossimo step

Una volta soddisfatti questi criteri, si procede alla **Fase 2 — Dataset Clean/Dirty**:
il modello YOLOv8 appena trainato viene usato per estrarre automaticamente i crop di finestre dalle immagini di facciate che vuoi etichettare come clean/dirty.

Vedere [pipeline_documentation.md](pipeline_documentation.md#fase-3--dataset-cleandirty) per i dettagli.

---

*Fase 1 — Training YOLOv8 Window Detector — Glass Cleanliness Detection — Aprile 2026*
