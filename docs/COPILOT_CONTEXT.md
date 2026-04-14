# Contesto Progetto — Glass Cleanliness Detection

Incolla questo file come primo messaggio in una nuova sessione GitHub Copilot per
dare tutto il contesto necessario.

---

## Descrizione del Progetto

Stiamo sviluppando un sistema di **Computer Vision per il controllo automatico della
pulizia delle superfici vetrate** (finestre) di edifici urbani, nell'ambito di una
challenge di Facility Management.

Il sistema riceve immagini riprese da **drone** che raffigurano facciate di edifici
e produce in output:
- Identificazione automatica di tutte le finestre (bounding box)
- Classificazione dello stato di pulizia per ogni finestra: **clean** o **dirty**
- **Heatmap visiva** (Grad-CAM) che mostra dove si concentra lo sporco
- Report strutturato (JSON) con coordinate e score per ogni pannello

---

## Architettura della Pipeline (2 modelli in serie)

```
Immagine drone (facciata edificio)
      ↓
[MODELLO 1 — YOLOv8]
  → rileva tutte le finestre → bounding box [x,y,w,h]
      ↓
  crop automatico di ogni finestra
      ↓
[MODELLO 2 — EfficientNet-B2]
  → classifica ogni crop: clean / dirty
  → Grad-CAM: heatmap dello sporco sulla finestra
      ↓
Output: immagine annotata + report JSON + heatmap
```

---

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Object Detection | YOLOv8n (ultralytics 8.4.37) |
| Classifier | EfficientNet-B2 (timm 1.0.26) |
| Explainability (heatmap) | Grad-CAM (pytorch-grad-cam) |
| Training tracking | MLflow |
| VM Training | Azure Standard_NV12ads_A10_v5 (GPU A10-8Q, 8GB VRAM, CUDA 12.8) |
| Storage | Azure Blob Storage (`stglasscleanliness`, RG `kpmg-bologna`) |
| VM IP | `20.9.194.236` — SSH: `azureuseram@20.9.194.236` |
| Conda env | `glass-cv` (Python 3.10, PyTorch 2.5.1+cu121) |
| Repo VM | `/mnt/project/AI-CHALLENGE-AM-BOLO/` |

---

## Stato Attuale (Aprile 2026)

### ✅ COMPLETATO — Fase 1: YOLOv8 Window Detector

Abbiamo eseguito **3 iterazioni di training** e scelto il modello finale:

| Versione | Dataset | Pesi partenza | Epoche | mAP50 | Esito |
|---|---|---|---|---|---|
| v1 | Roboflow (3866 img) | yolov8n.pt (COCO) | 100 | 0.984 | Buono ma bias su finestre ravvicinate |
| v2 | Merged (7281 img) | yolov8_best.pt (v1) | 21 (early stop) | 0.963 | ❌ Distribution shift |
| **v3** | **Merged (7281 img)** | **yolov8n.pt (COCO)** | **100** | **0.947** | ✅ **Scelto** |

**Lesson learned:** il continual fine-tuning (v2) ha fallito per distribution shift —
il modello partiva già specializzato su finestre ravvicinate (Roboflow) e non riusciva
ad adattarsi alle facciate intere (glasswindow). La soluzione è stata ripartire da
pesi COCO generici sul dataset merged.

**Dataset usati per YOLO:**
1. `window-detection-vnpow-v1` (Roboflow) — finestre ravvicinate annotate, formato YOLOv8 txt
2. `monymon/dataset_glasswindow` (GitHub) — facciate intere, 6 classi originali, filtrata solo classe 0 (glass_window)

Script di merge: `src/data/merge_datasets.py`
Dataset merged su VM: `/mnt/project/data/raw/merged_yolov8/`

**Modello finale su Blob Storage:** `stglasscleanliness/models/yolov8_windows_v3_merged_best.pt`
**Modello locale:** `models/weights/yolov8_windows_v3_merged_best.pt`

---

### 🔄 IN CORSO — Fase 2: EfficientNet Cleanliness Classifier

**Dataset disponibile:** ~200 immagini clean + ~200 immagini dirty (già raccolte)

**Pipeline prevista per costruire il dataset EfficientNet:**
1. `src/data/filter_and_crop_windows.py` — usa YOLOv8 su facciate non etichettate,
   croppa finestre con conf >= 0.70, scarica in `data/windows/to_label/`
2. `src/data/label_windows.py` — labeling interattivo (tasto C/D/S per clean/dirty/skip)
3. `src/data/build_efficientnet_dataset.py` — combina crop labellati + 200+200 originali,
   split 70/15/15 in `data/windows_efficientnet/`

**Script di training:** `src/training/train_efficientnet.py` (da creare)
**Architettura:** EfficientNet-B2, pretrained ImageNet, fine-tuning su 2 classi
**Heatmap:** Grad-CAM su layer finale del backbone (no training extra richiesto)

---

## Struttura del Progetto

```
AI-CHALLENGE2/
├── docs/
│   ├── pipeline_documentation.md      ← pipeline completa (6 fasi)
│   ├── phase0_infrastructure_setup.md ← setup VM Azure, azcopy, conda
│   ├── phase1_yolo_training.md        ← training YOLOv8 (3 iterazioni documentate)
│   └── COPILOT_CONTEXT.md             ← questo file
├── src/
│   ├── data/
│   │   ├── merge_datasets.py          ← merge Roboflow + glasswindow
│   │   ├── filter_and_crop_windows.py ← YOLOv8 → crop finestre da facciate
│   │   ├── label_windows.py           ← labeling interattivo C/D/S
│   │   └── build_efficientnet_dataset.py ← costruisce dataset finale
│   ├── training/
│   │   └── train_yolo.py              ← training YOLOv8 (--model, --data, --epochs)
│   └── inference/                     ← da creare
├── configs/
│   ├── model.yaml                     ← architetture (yolov8n + efficientnet_b2)
│   └── training.yaml                  ← iperparametri (yolo: + efficientnet:)
├── data/
│   ├── window_detection.yaml          ← dataset config Roboflow (percorsi VM)
│   ├── merged_window_detection.yaml   ← dataset config merged (percorsi VM)
│   ├── raw/roboflow/yolov8/           ← dataset Roboflow formato YOLOv8
│   └── windows/to_label/              ← crop da etichettare (Fase 2)
├── models/weights/
│   ├── yolov8_windows_best.pt               ← v1 (mAP50=0.984)
│   ├── yolov8_windows_merged_best.pt        ← v2 (mAP50=0.963, scartato)
│   └── yolov8_windows_v3_merged_best.pt     ← v3 (mAP50=0.947) ← USARE QUESTO
└── am_images_test/
    └── testDetection.py               ← confronto visivo v1/v2/v3
```

---

## Note Operative Importanti

**Azure azcopy v10 su Windows:**
- Non ha `--account-key` flag → generare SAS token e metterlo nell'URL
- Comando: `az storage container generate-sas --account-key "$key"` → embed in URL

**azcopy snap su VM:**
- Sandbox snap non accede a file root-owned o `/tmp/`
- Per upload da VM usare sempre `az storage blob upload` (non azcopy)

**ultralytics MLflow:**
- Ha integrazione interna che confligge → `yolo settings mlflow=False` prima di trainare

**Dataset formato:**
- ultralytics richiede formato YOLOv8 txt (non COCO JSON)
- Download Roboflow: `project.version(1).download('yolov8', ...)`

**Best.pt dopo training:**
- I file sono root-owned → `sudo cp best.pt ~/` + `sudo chmod 644` prima di uploadare

**MLflow non raggiungibile da fuori VM:**
- SSH tunnel: `ssh -L 5000:localhost:5000 azureuseram@20.9.194.236`

---

## Prossimi Step

1. **Fase 2a** — Preparare dataset EfficientNet:
   - Lanciare `filter_and_crop_windows.py` su facciate non etichettate
   - Eseguire `label_windows.py` per etichettare i crop
   - Eseguire `build_efficientnet_dataset.py`

2. **Fase 2b** — Creare e lanciare `src/training/train_efficientnet.py`
   - EfficientNet-B2, pretrained ImageNet, fine-tuning 2 classi
   - Target: val accuracy > 85%, F1-dirty > 80%

3. **Fase 3** — Pipeline inferenza end-to-end (`src/inference/`)
   - YOLOv8 → crop → EfficientNet → Grad-CAM heatmap → report JSON

4. **Fase 4** — Test su facciate Bologna reali
