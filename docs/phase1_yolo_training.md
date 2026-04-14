# Fase 1 — Training YOLOv8 Window Detector
## Documentazione Tecnica e Funzionale

**Progetto:** Glass Cleanliness Detection — AI Computer Vision  
**Step:** 1 di 6 — Rilevamento automatico delle finestre nelle facciate  
**Data:** Aprile 2026  
**Stato:** ✅ Fase 1 completa — 3 training eseguiti, modello finale scelto: **v3 merged from scratch** (mAP50=0.947, più robusto su facciate reali)  
**Prerequisito:** Fase 0 completata (VM online, conda `glass-cv` attivo, GPU A10-8Q disponibile)

---

## Indice

1. [Obiettivo](#1-obiettivo)
2. [Chiarimento: COCO Dataset vs Formato vs Pesi](#2-chiarimento-coco-dataset-vs-formato-vs-pesi)
3. [Come Funziona l Addestramento YOLOv8](#3-come-funziona-laddestramento-yolov8)
4. [Il Dataset: window-detection-vnpow](#4-il-dataset-window-detection-vnpow)
5. [Setup su VM - Passi Eseguiti](#5-setup-su-vm--passi-eseguiti)
6. [Script di Training](#6-script-di-training)
7. [Esecuzione del Training](#7-esecuzione-del-training)
8. [Risultati Smoke Test 1 epoca](#8-risultati-smoke-test-1-epoca)
9. [Cosa Caricare su Storage per Test Locale](#9-cosa-caricare-su-storage-per-test-locale)
10. [Analisi Risultati Training Completo](#10-analisi-risultati-training-completo)
11. [Iterazioni di Training e Scelta Finale](#11-iterazioni-di-training-e-scelta-finale)
12. [Criteri di Completamento](#12-criteri-di-completamento)

---

## 1. Obiettivo

Addestrare un modello YOLOv8 capace di localizzare tutte le finestre presenti in un immagine di facciata di edificio, producendo per ognuna un **bounding box** con relativo **confidence score**.

```
Immagine facciata
      |
      v
 [YOLOv8]  <--- addestriamo in questa fase
      |
      v
bbox di ogni finestra -> crop -> [EfficientNet] -> clean / dirty
```

**Object Detection vs Bounding Box:** non sono due cose separate. Il bounding box e l output dell object detection. YOLO produce bounding box — rettangoli `(x, y, w, h)` attorno a ogni finestra trovata. **Non produce maschere pixel-per-pixel** (quella sarebbe segmentazione, rimossa dalla pipeline).

---

## 2. Chiarimento: COCO Dataset vs Formato vs Pesi

| Cosa | Descrizione | Nel progetto |
|---|---|---|
| **Dataset COCO** (Microsoft) | 330k immagini, 80 classi (persone, auto...) | No — non usato |
| **Formato COCO** (JSON) | Standard per distribuire annotazioni bbox | Si — dataset Roboflow in COCO format |
| **`yolov8n.pt`** | Parametri rete pre-addestrata su COCO | Si — punto di partenza fine-tuning |
| **`window-detection-vnpow`** | Il nostro dataset (finestre in facciate) | Si — training supervisionato |

> Roboflow esporta in formato COCO per compatibilita, ma noi abbiamo scaricato il dataset in **formato YOLOv8 nativo** (file `.txt` per label) che ultralytics legge direttamente senza conversione.

---

## 3. Come Funziona l Addestramento YOLOv8

### 3.0 Transfer Learning e Fine-Tuning

Un modello di deep learning è una struttura matematica (architettura) i cui parametri
interni si chiamano **pesi**. Senza pesi il modello è "vuoto" e non sa fare nulla.

Il training è il processo che ottimizza questi pesi su un dataset specifico.
Invece di partire da zero (pesi casuali), utilizziamo il **transfer learning**:

```
1. Architettura YOLOv8n (pesi casuali — modello vuoto)
         |
         v
2. Ultralytics traine su COCO 2017
   (80 classi, 118k immagini, milioni di oggetti)
         |
         v
3. yolov8n.pt  ← pesi pretrained  (questo scarichiamo da ultralytics)
         |
         v  fine-tuning: ottimizziamo su finestre
4. window-detection dataset (~4800 immagini, 1 classe)
         |
         v
5. yolov8_windows_best.pt  ← i NOSTRI pesi  (best.pt su Blob Storage)
```

**Perché funziona con "pochi" dati:**
il modello non parte da zero — sa già riconoscere bordi, texture, forme geometriche
grazie a COCO. Il fine-tuning aggiusta solo lo strato finale per specializzarsi
sulle finestre. Risultato: mAP50=0.984 con ~4800 immagini.

**A inferenza si caricano sempre i propri pesi:**
```python
model = YOLO("models/weights/yolov8_windows_best.pt")
# Non si scarica niente da internet — i pesi sul disco contengono già tutto
```

Lo stesso pattern si ripete per EfficientNet (Fase 2):
pesi ImageNet → fine-tuning su clean/dirty → efficientnet_best.pt

---

### 3.1 Architettura (yolov8n: 130 layer, 3.0M parametri, 8.2 GFLOPs)

```
Input (640x640 px)
      |
      v
BACKBONE (CSPDarknet)  - Estrae feature: bordi, texture, forme
                       - Pre-addestrato su COCO, aggiornato poco
      |
      v
NECK (PAN-FPN)         - Combina feature a scale diverse
                       - Per rilevare finestre grandi e piccole
      |
      v
HEAD (Detect)          - Predice: bbox + confidence + classe
  751k params          - Si specializza completamente nel fine-tuning
      |
      v
Lista di [x, y, w, h, conf, classe] per ogni finestra trovata
```

### 3.2 Metriche di Valutazione

| Metrica | Significato | Target |
|---|---|---|
| **mAP50** | Mean Average Precision a IoU >= 0.50 | > 0.75 |
| **mAP50-95** | Media mAP da IoU 0.50 a 0.95 | > 0.55 |
| **Precision** | Frazione di box predetti corretti | > 0.80 |
| **Recall** | Frazione di finestre reali trovate | > 0.75 |

IoU = area intersezione / area unione tra box predetto e ground truth.

### 3.3 Augmentation Configurata

| Parametro | Valore | Motivazione |
|---|---|---|
| `fliplr=0.5` | Flip orizzontale 50% | Edifici simmetrici |
| `flipud=0.0` | No flip verticale | Finestre hanno orientamento fisso |
| `degrees=5.0` | Rotazione +/-5 gradi | Drone leggermente inclinato |
| `hsv_v=0.4` | Variazione luminosita | Ore del giorno diverse |
| `scale=0.5` | Zoom +/-50% | Distanza variabile dal drone |
| `mosaic=1.0` | Unisce 4 immagini in 1 | Aumenta varieta di contesti |

---

## 4. Il Dataset: window-detection-vnpow

### Provenienza
Dataset pubblico: https://universe.roboflow.com/research-xvh79/window-detection-vnpow  
Workspace: `research-xvh79`, project: `window-detection-vnpow`, version: `1`

### Statistiche

| Split | Immagini | Label (window) |
|-------|----------|----------------|
| Train | 3.866 | ~15.000 bbox |
| Valid | 484 | ~1.800 bbox |
| Test | 490 | ~1.900 bbox |
| **Totale** | **4.840** | **~18.700 bbox** |

### Formato Scaricato: YOLOv8

```
data/raw/roboflow/yolov8/
├── data.yaml              <- config con path e nomi classi (generato da Roboflow)
├── train/
│   ├── images/            <- immagini .jpg
│   └── labels/            <- file .txt, uno per immagine
│       └── facade_001.txt <- ogni riga = una finestra: "0 cx cy w h" (normalizzati)
├── valid/
└── test/
```

Formato label YOLOv8 (ogni riga in .txt):
```
0 0.453 0.312 0.156 0.234
^ ^     ^     ^     ^
cls cx  cy    w     h   (tutti 0-1 normalizzati rispetto dimensioni immagine)
```

### Download sulla VM

```bash
conda activate glass-cv
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='<ROBOFLOW_API_KEY>')
project = rf.workspace('research-xvh79').project('window-detection-vnpow')
project.version(1).download('yolov8', location='/mnt/project/data/raw/roboflow/yolov8')
"
```

---

## 5. Setup su VM — Passi Eseguiti

### 5.1 Struttura progetto su VM

```
/mnt/project/
├── AI-CHALLENGE-AM-BOLO/   <- repo git clonato (codice + config)
│   ├── src/training/train_yolo.py
│   ├── data/window_detection.yaml
│   ├── configs/training.yaml
│   └── configs/model.yaml
├── data/
│   └── raw/roboflow/
│       └── yolov8/         <- dataset formato YOLOv8 usato per training
│           ├── train/images/ + train/labels/
│           ├── valid/images/ + valid/labels/
│           └── test/images/  + test/labels/
├── runs/                   <- output training
└── logs/
```

### 5.2 MLflow Avviato

```bash
conda activate glass-cv
mkdir -p /mnt/mlflow-data/{mlruns,artifacts}
nohup mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri /mnt/mlflow-data/mlruns \
  --default-artifact-root /mnt/mlflow-data/artifacts \
  > /mnt/mlflow-data/mlflow.log 2>&1 &
```

MLflow UI: `http://20.9.194.236:5000`

### 5.3 Problema Risolto: MLflow Callback Ultralytics

Ultralytics ha una propria integrazione MLflow interna che interferisce con la nostra.  
Soluzione:

```bash
yolo settings mlflow=False
```

---

## 6. Script di Training

File: `src/training/train_yolo.py`

Parametri chiave (da `configs/training.yaml`, sezione `yolo`):

```yaml
epochs: 100
batch_size: 16    # A10-8Q (8GB VRAM): batch 16 usa ~2.4GB -> ampio margine
img_size: 640
device: 0
patience: 20      # early stopping se mAP non migliora per 20 epoche
optimizer: SGD
lr0: 0.01
cos_lr: true      # cosine LR decay
amp: true         # mixed precision -> dimezza uso VRAM
```

---

## 7. Esecuzione del Training

### Smoke test (1 epoca — verifica che tutto funzioni)

```bash
cd /mnt/project/AI-CHALLENGE-AM-BOLO
conda activate glass-cv
python src/training/train_yolo.py --epochs 1 --device 0
```

### Training completo (background, chiudi pure SSH)

```bash
nohup python src/training/train_yolo.py --epochs 100 --device 0 \
  > runs/train_yolo_full.log 2>&1 &
echo "Training PID: $!"
```

### Monitoraggio

```bash
# Log epoche
tail -f runs/train_yolo_full.log

# GPU usage
watch -n 5 nvidia-smi

# MLflow UI
xdg-open http://localhost:5000   # oppure dal browser locale: http://20.9.194.236:5000
```

### Tempi stimati su A10-8Q

| Variante | Epoche | Tempo |
|---|---|---|
| yolov8n | 100 | ~90 minuti |
| yolov8s | 100 | ~150 minuti |
| yolov8m | 100 | ~240 minuti |

---

## 8. Risultati Smoke Test (1 epoca)

Eseguito il 13 Aprile 2026 su VM `BOLOGNA-AI-AM-MACHINE` (A10-8Q):

| Metrica | Risultato | Note |
|---|---|---|
| **mAP50** | **0.460** | Ottimo per 1 sola epoca |
| **mAP50-95** | **0.256** | |
| **Precision** | 0.456 | |
| **Recall** | 0.544 | |
| GPU memory usata | 2.42 GB / 8 GB | Ampio margine |
| Tempo per epoca | ~55 secondi | 100 epoche = ~90 min |
| Velocita inferenza | 2.7ms/immagine | |
| Batch size | 16 @ 640px | |

Output salvati in:
```
runs/window_detection/yolov8n_windows/
├── weights/
│   ├── best.pt     <- modello con miglior mAP50
│   └── last.pt     <- modello all ultima epoca
├── results.png     <- curve loss e metriche per epoca
├── PR_curve.png    <- precision-recall curve
├── confusion_matrix.png
└── labels.jpg      <- visualizzazione distribuzione label
```

Run MLflow: `http://20.9.194.236:5000/#/experiments/849287110222805549`

---

## 9. Cosa Caricare su Storage per Test Locale

Per testare il modello trainato in locale (Windows) serve **solo il file `best.pt`** (6.3 MB).

---

### Upload dalla VM → Azure Blob

> **Problema riscontrato:** azcopy sulla VM (installato via snap) non riesce ad accedere
> a file fuori dalla home dell'utente per via della sandbox snap. Soluzione: copiare
> il file nella home prima di uploadarlo, poi eliminare la copia.

```bash
# Sulla VM — procedura corretta
sudo cp runs/window_detection/yolov8n_windows/weights/best.pt ~/yolov8_best.pt \
&& sudo chmod 644 ~/yolov8_best.pt \
&& key=$(az storage account keys list \
  --account-name stglasscleanliness \
  --resource-group kpmg-bologna \
  --query "[0].value" -o tsv) \
&& az storage blob upload \
  --account-name stglasscleanliness \
  --container-name models \
  --name yolov8_windows_best.pt \
  --file ~/yolov8_best.pt \
  --account-key "$key" \
  --overwrite \
&& rm ~/yolov8_best.pt
```

> **Nota:** si usa `az storage blob upload` (non azcopy) perché az non ha i problemi
> di sandbox snap. Per upload di cartelle intere usare `az storage blob upload-batch`.

---

### Download Storage → PC locale (Git Bash su Windows)

```bash
key=$(az storage account keys list \
  --account-name stglasscleanliness \
  --resource-group kpmg-bologna \
  --query "[0].value" -o tsv) \
&& sas=$(az storage container generate-sas \
  --account-name stglasscleanliness --name models \
  --permissions rl --expiry 2026-12-31T23:59:00Z \
  --account-key "$key" --output tsv) \
&& "$USERPROFILE/bin/azcopy.exe" copy \
  "https://stglasscleanliness.blob.core.windows.net/models/yolov8_windows_best.pt?${sas}" \
  "C:/Users/andreamancini/Downloads/AI-CHALLENGE2/models/weights/yolov8_windows_best.pt"
```

File scaricato: `models/weights/yolov8_windows_best.pt` (6.3 MB) ✅

---

### Test del modello in locale (inferenza)

> **Cos'è l'inferenza:** dopo il training (fase in cui la rete impara), l'inferenza è
> l'applicazione del modello su dati nuovi per ottenere predizioni. Nessun peso viene
> aggiornato — si usano solo quelli già appresi.

**Dove eseguire:** in locale su Windows, dalla cartella del progetto in PowerShell.
`ultralytics` è già installato nell'ambiente Python locale.

```python
# Esecuzione: cd AI-CHALLENGE2 && python scripts/test_model.py
from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/weights/yolov8_windows_best.pt")

# Su una singola immagine di facciata
results = model.predict(
    source="data/facades/tua_facciata.jpg",
    conf=0.25,      # confidence threshold: ignora box con confidenza < 25%
    iou=0.45,       # NMS: rimuove box sovrapposti con IoU > 45%
    save=True,      # salva immagine con box disegnati
    project="runs/inference",
    name="test_locale"
)

# Stampa i bounding box trovati
for r in results:
    print(f"Finestre trovate: {len(r.boxes)}")
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        print(f"  ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})  conf={conf:.2f}")

# Output: immagine annotata in runs/inference/test_locale/
```

Su una cartella di facciate:
```python
results = model.predict(
    source="data/facades/",    # processa tutte le immagini nella cartella
    conf=0.25,
    save=True,
    project="runs/inference",
    name="bologna_batch"
)
```

---

## 10. Analisi Risultati Training Completo

Training eseguito il 13 Aprile 2026 — 100 epoche in **1h 35min** su A10-8Q.

| Metrica | Target | Risultato | Esito |
|---|---|---|---|
| **mAP50** | > 0.75 | **0.984** | ✅ Superato |
| **mAP50-95** | > 0.55 | **0.855** | ✅ Superato |
| **Precision** | > 0.80 | **0.954** | ✅ Superato |
| **Recall** | > 0.75 | **0.954** | ✅ Superato |

Il modello individua il 95.4% delle finestre reali con il 95.4% di precisione.  
Ottimo per l'uso in Fase 2 (crop finestre da facciate nuove).

MLflow run: `http://20.9.194.236:5000/#/experiments/849287110222805549/runs/0a3d0525777c424a89ccb6187c4b8038`

> **Nota:** MLflow UI raggiungibile solo dalla rete locale della VM o con port forwarding SSH:
> ```bash
> ssh -L 5000:localhost:5000 azureuseram@20.9.194.236
> # poi apri http://localhost:5000 nel browser locale
> ```

---

## 11. Iterazioni di Training e Scelta Finale

### Panoramica delle 3 iterazioni

| | v1 | v2 | v3 ✅ scelto |
|---|---|---|---|
| **Pesi partenza** | yolov8n.pt (COCO) | yolov8_best.pt (v1) | yolov8n.pt (COCO) |
| **Dataset** | Roboflow only (3866 train) | Merged (7281 train) | Merged (7281 train) |
| **Epoche** | 100 | 21 (early stop) | 100 |
| **mAP50** | 0.984 | 0.963 | 0.947 |
| **mAP50-95** | 0.855 | 0.804 | 0.770 |
| **Precision** | 0.954 | 0.954 | 0.932 |
| **Recall** | 0.954 | 0.893 | 0.884 |
| **Durata** | 1h 35min | 43min | 3h 24min |
| **Blob Storage** | `yolov8_windows_best.pt` | `yolov8_windows_merged_best.pt` | `yolov8_windows_v3_merged_best.pt` |

---

### Training v1 — Roboflow only (13 Aprile 2026)

Primo training su dataset Roboflow puro (~4840 immagini di finestre ravvicinate).
mAP50=0.984 sul validation set Roboflow. Ottimo in assoluto, ma validation set
e training set hanno la stessa distribuzione → rischio overfitting alla distribuzione
"finestre ravvicinate".

---

### Training v2 — Continual Fine-tuning (14 Aprile 2026) ❌ scartato

**Motivazione:** aggiungere il dataset `monymon/dataset_glasswindow` (facciate intere)
per migliorare la robustezza. Partendo dai pesi v1 già ottimizzati si sperava di
convergere in meno epoche.

**Problema: Distribution Shift**
Il modello v1 era "convinto" della distribuzione Roboflow (finestre ravvicinate, alta
densità di label). Le facciate intere del glasswindow hanno finestre più piccole,
più sparse, con più contesto. Il modello non è riuscito a riconciliare le due
distribuzioni e l'early stopping si è attivato alla epoch 21 (best all'epoch 1).

Confermato empiricamente: test visivo su facciata reale → v2 più impreciso di v1.

---

### Training v3 — Merged from Scratch (14 Aprile 2026) ✅ scelto

**Soluzione al distribution shift:** ricominciare dai pesi COCO generici (`yolov8n.pt`)
che non hanno bias verso nessuna delle due distribuzioni. Il modello impara entrambe
le distribuzioni contemporaneamente durante le 100 epoche.

**mAP50 leggermente inferiore a v1** (0.947 vs 0.984) perché il validation set è
ancora solo Roboflow (484 immagini ravvicinate) — ma il modello è più robusto su
facciate reali come confermato dal test visivo.

**Test visivo comparativo** (`am_images_test/testDetection.py`):
- v1: alta precision su finestre ravvicinate, qualche miss su facciate intere
- v2: impreciso, bounding box errati su facciate reali
- v3: migliore equilibrio — trova più finestre correttamente su facciate intere ✅

**Modello scelto per la Fase 2:** `models/weights/yolov8_windows_v3_merged_best.pt`

---

### Script di confronto usato per la scelta

```python
# am_images_test/testDetection.py
from ultralytics import YOLO

IMAGE = "am_images_test/AMI1.png"

models = {
    "v1_roboflow":       ("models/weights/yolov8_windows_best.pt",           "mAP50=0.984"),
    "v2_continual_ft":   ("models/weights/yolov8_windows_merged_best.pt",    "mAP50=0.963"),
    "v3_merged_scratch": ("models/weights/yolov8_windows_v3_merged_best.pt", "mAP50=0.947"),
}

for name, (weights, desc) in models.items():
    model = YOLO(weights)
    results = model.predict(IMAGE, conf=0.25, save=True, project="runs/inference", name=name)
    print(f"[{name}] {desc} → {len(results[0].boxes)} finestre trovate")
```

Output annotati in `runs/inference/v1/`, `v2/`, `v3/`.

---

## 12. Re-Training con Dataset Esteso (storico — vedere Sezione 11)

> Questa sezione documenta la cronologia — per il riepilogo finale vedere Sezione 11.

### Dataset Aggiunto: monymon/dataset_glasswindow

**NOTA STORICA — re-training

### Motivazione

Il primo training (100 epoche, mAP50=0.984) era su solo ~4800 immagini di finestre
ritagliate ravvicinate. Per migliorare la robustezza su **facciate intere** (contesti diversi,
scale diverse, finestre parzialmente visibili) è stato aggiunto un secondo dataset.

### Dataset Aggiunto: monymon/dataset_glasswindow

- **Fonte:** https://github.com/monymon/dataset_glasswindow
- **Caratteristiche:** facciate di edifici intere (non solo finestre ravvicinate), con
  annotazioni in formato YOLOv8 txt
- **Classi originali:** 6 — `glass_window(0)`, pillar(1), wall(2), person(3), door(4), car(5)
- **Classi usate:** solo classe 0 (`glass_window`) → rimappata a `window(0)` del nostro schema
- **Immagini utili** (con almeno una finestra): da verificare a fine merge

### Merge dei Dataset

Script: `src/data/merge_datasets.py`

Il merge filtra le label del dataset glasswindow tenendo solo `class_id=0`, scarta
le immagini senza finestre, rinomina i file con prefisso `rf_` (Roboflow) e `gw_`
(glasswindow) per evitare conflitti.

```bash
# In locale
git clone https://github.com/monymon/dataset_glasswindow.git data/raw/dataset_glasswindow
python src/data/merge_datasets.py
# → output: data/raw/merged_yolov8/ + data/merged_window_detection.yaml
```

Upload su Blob e download su VM:
```bash
# Su VM — download da Blob (--source = container, --pattern = prefisso)
az storage blob download-batch \
  --account-name stglasscleanliness \
  --source datasets \
  --destination /mnt/project/data/raw/ \
  --pattern "merged_yolov8/*" \
  --account-key "$key"
```

### Fine-Tuning (Continual Learning)

Il re-training parte dai pesi `yolov8_windows_best.pt` (già fine-tuned, mAP50=0.984)
invece di `yolov8n.pt` (pretrained COCO). Questo si chiama **continual fine-tuning**:
il modello non riparte da zero ma affina la conoscenza già acquisita.

```bash
# Sulla VM — avviato il 14 Aprile 2026
nohup python src/training/train_yolo.py \
  --model models/weights/yolov8_windows_best.pt \
  --data  data/merged_window_detection.yaml \
  --epochs 50 \
  --device 0 \
  > runs/train_yolo_merged.log 2>&1 &
```

**Perché 50 epoche invece di 100:** il modello parte già da mAP50≈0.984, la
convergenza sul dataset esteso richiede meno iterazioni.

**Stato:** 🔄 Training in corso — risultati da aggiornare al termine.

---

## 13. Criteri di Completamento

- [x] Dataset `window-detection-vnpow-v1` scaricato in formato YOLOv8 sulla VM
- [x] `data/window_detection.yaml` configurato correttamente
- [x] MLflow server avviato (`http://20.9.194.236:5000`)
- [x] Script `src/training/train_yolo.py` funzionante (aggiornato per accettare `--model path.pt` e `--data`)
- [x] Smoke test 1 epoca completato (mAP50=0.46, GPU 2.4GB/8GB)
- [x] Training v1 — 100 epoche, Roboflow only (mAP50=0.984, 1h 35min)
- [x] Training v2 — 21 epoche, continual fine-tuning (mAP50=0.963, scartato per distribution shift)
- [x] Training v3 — 100 epoche, merged from scratch (mAP50=0.947, 3h 24min) ✅ **scelto**
- [x] Tutti e 3 i modelli su Azure Blob Storage (`models/`)
- [x] Tutti e 3 i modelli scaricati in locale (`models/weights/`)
- [x] Test visivo comparativo completato — v3 scelto come modello finale
- [x] Dataset glasswindow mergiato con Roboflow (`src/data/merge_datasets.py`)
- [ ] Fase 2: training EfficientNet clean/dirty con dataset 200+200 immagini

---

## Prossimi Step (dopo training completo)

```bash
# 1. Verifica metriche finali
tail -20 runs/train_yolo_full.log

# 2. Carica best.pt su Storage (vedi Sezione 9)

# 3. Scarica e testa in locale (vedi Sezione 9)

# 4. Fase 2: usa YOLO per estrarre finestre dalle facciate Bologna
#    Output: data/windows/{train,valid,test}/{clean,dirty}/
#    Vedi: docs/pipeline_documentation.md — Fase 2
```
