az login --use-device-code
echo 'export PATH="$PATH:/c/Users/andreamancini/AppData/Roaming/Python/Python311/Scripts"' >> ~/.bashrc
source ~/.bashrc
# Glass Cleanliness Detection — Documentazione Completa della Pipeline

**Progetto:** AI Computer Vision per Facility Management — Analisi Pulizia Superfici Vetrate  
**Versione:** 1.0  
**Data:** Aprile 2026

---

## Indice

1. [Panoramica del Sistema](#1-panoramica-del-sistema)
2. [Architettura Generale](#2-architettura-generale)
3. [FASE 0 — Setup Infrastruttura Azure](#fase-0--setup-infrastruttura-azure)
4. [FASE 1 — Raccolta e Preparazione Immagini](#fase-1--raccolta-e-preparazione-immagini)
5. [FASE 2 — Segmentazione Automatica delle Vetrate](#fase-2--segmentazione-automatica-delle-vetrate)
6. [FASE 3 — Human Refinement con CVAT](#fase-3--human-refinement-con-cvat)
7. [FASE 4 — AI-Assisted Labeling della Pulizia](#fase-4--ai-assisted-labeling-della-pulizia)
8. [FASE 5 — Training del Modello Finale](#fase-5--training-del-modello-finale)
9. [FASE 6 — Heatmap e Output Visivo](#fase-6--heatmap-e-output-visivo)
10. [Stack Tecnologico](#stack-tecnologico)
11. [Struttura del Repository](#struttura-del-repository)

---

## 1. Panoramica del Sistema

### Cosa fa questo sistema?

Il sistema riceve in input **immagini o video di facciate di edifici** acquisiti da drone e produce in output:

- La **localizzazione** di ogni pannello vetrato nell'immagine
- Una **valutazione dello stato di pulizia** per ogni pannello (da "pulito" a "molto sporco")
- Una **heatmap visiva** sovrapposta all'immagine che evidenzia le zone sporche
- Un **report numerico** con coordinate e score di pulizia per ogni vetrata

### A chi serve?

- **Facility manager**: per pianificare interventi di pulizia mirati, evitando ispezioni manuali costose
- **Operatori drone**: per sapere quali sezioni dell'edificio richiedono attenzione
- **Team tecnico**: per integrare il sistema in piattaforme di building management (BMS)

### Perché AI Computer Vision?

Un drone acquisisce centinaia di immagini per edificio. Valutare manualmente ogni vetrata è impossibile su scala. Il modello AI automatizza questa analisi in pochi secondi per immagine, con consistenza e ripetibilità garantite.

---

## 2. Architettura Generale

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT                                      │
│          Immagini/Video drone di facciate edifici                │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MODULO 1: SEGMENTAZIONE                        │
│   Grounding DINO → rileva dove sono le vetrate (bounding box)    │
│   SAM2           → ritaglia la forma esatta di ogni vetrata      │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  MODULO 2: CLASSIFICAZIONE PULIZIA               │
│   EfficientNet-B2 → per ogni vetrata segmentata,                 │
│                     predice il livello di sporcizia (0-3)        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MODULO 3: HEATMAP                              │
│   Grad-CAM → genera mappa di calore che mostra DOVE è lo sporco  │
│              all'interno di ogni pannello vetrato                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                     │
│   - Immagine annotata con heatmap colorata                       │
│   - JSON report (coordinate + score per vetrata)                 │
│   - Dashboard opzionale (Streamlit / Power BI)                   │
└──────────────────────────────────────────────────────────────────┘
```

**Livelli di sporcizia (classi del modello):**

| Classe | Label | Descrizione |
|--------|-------|-------------|
| 0 | Clean | Vetro pulito, nessun deposito visibile |
| 1 | Slightly Dirty | Polvere leggera, quasi invisibile da vicino |
| 2 | Moderately Dirty | Depositi visibili, colature d'acqua, macchie |
| 3 | Heavily Dirty | Guano, alghe, sporco pesante e localizzato |

---

## FASE 0 — Setup Infrastruttura Azure

> **Obiettivo funzionale:** Preparare l'ambiente cloud dove girerà tutto il progetto: storage per le immagini, macchina GPU per il training, e strumenti di annotazione.

> **Chi esegue questa fase:** Team tecnico / DevOps

---

### 0.1 Risorse Azure da creare

#### A) Resource Group
Un "contenitore logico" che raggruppa tutte le risorse del progetto.

```bash
az group create \
  --name rg-glass-cleanliness \
  --location italynorth
```

#### B) Storage Account + Container Blob
Dove vengono salvate tutte le immagini raw, le annotazioni e i modelli addestrati.

```bash
# Crea Storage Account
az storage account create \
  --name stglasscleanliness \
  --resource-group rg-glass-cleanliness \
  --location italynorth \
  --sku Standard_LRS

# Crea i container (cartelle) principali
az storage container create --name raw-images       --account-name stglasscleanliness
az storage container create --name annotations      --account-name stglasscleanliness
az storage container create --name processed        --account-name stglasscleanliness
az storage container create --name models           --account-name stglasscleanliness
az storage container create --name outputs          --account-name stglasscleanliness
```

**Struttura Blob Storage:**
```
stglasscleanliness/
├── raw-images/          ← immagini originali drone
│   ├── building_001/
│   ├── building_002/
│   └── ...
├── annotations/         ← file JSON con maschere segmentazione
├── processed/           ← patch di vetrate ritagliate + labeled
├── models/              ← checkpoint .pt e modelli ONNX
└── outputs/             ← heatmap e report finali
```

#### C) Azure ML Workspace
Ambiente per tracciare esperimenti di training (tiene traccia di metriche, versioni modelli, parametri).

```bash
az ml workspace create \
  --name aml-glass-cleanliness \
  --resource-group rg-glass-cleanliness \
  --location italynorth
```

#### D) VM con GPU per Training

**Perché una GPU?** Il training di reti neurali senza GPU impiegherebbe giorni invece di ore.

```bash
# VM Standard_NC24ads_A100_v4 = 24 vCPU, 220GB RAM, 1x NVIDIA A100 80GB
az vm create \
  --resource-group rg-glass-cleanliness \
  --name vm-training-gpu \
  --image Ubuntu2204 \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --os-disk-size-gb 256
```

#### E) CVAT — Tool di Annotazione (via Docker)

CVAT è uno strumento open source per annotare immagini. Lo eseguiamo sulla stessa VM GPU.

```bash
# Connettiti alla VM
ssh azureuser@<VM_PUBLIC_IP>

# Installa Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Scarica e avvia CVAT
git clone https://github.com/opencv/cvat.git
cd cvat
docker compose up -d

# CVAT sarà accessibile su http://<VM_IP>:8080
```

> **Nota funzionale:** CVAT è come un "Photoshop collaborativo" per annotazioni AI. Gli annotatori aprono le immagini nel browser, vedono le maschere generate automaticamente da SAM2 e le correggono dove necessario.

---

### 0.2 Setup Ambiente Python sulla VM

```bash
# Installa Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Crea ambiente del progetto
conda create -n glass-cv python=3.11
conda activate glass-cv

# Installa dipendenze CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Installa librerie progetto
pip install \
  ultralytics \          # YOLOv8
  segment-anything-2 \   # SAM2
  transformers \          # Grounding DINO
  supervision \           # utilities annotazione
  albumentations \        # data augmentation
  pytorch-lightning \     # training framework
  mlflow \                # experiment tracking
  grad-cam \              # heatmap Grad-CAM
  opencv-python \
  Pillow \
  pandas \
  matplotlib \
  azure-storage-blob \
  azure-ai-ml
```

---

### 0.3 Struttura del Repository

```
glass-cleanliness-detection/
│
├── data/
│   ├── raw/                    ← immagini originali (sincronizzato con Blob)
│   ├── annotations/            ← COCO JSON con maschere vetrate
│   ├── patches/                ← patch ritagliate per classificatore
│   └── splits/                 ← train/val/test split CSV
│
├── src/
│   ├── phase1_collection/
│   │   ├── scrape_mapillary.py
│   │   └── quality_filter.py
│   │
│   ├── phase2_segmentation/
│   │   ├── grounding_dino_detect.py
│   │   ├── sam2_segment.py
│   │   └── export_coco.py
│   │
│   ├── phase4_labeling/
│   │   ├── clip_zero_shot.py
│   │   ├── active_learning.py
│   │   └── feature_extractor.py
│   │
│   ├── phase5_training/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── phase6_inference/
│       ├── pipeline.py
│       ├── gradcam_heatmap.py
│       └── report_generator.py
│
├── configs/
│   ├── azure.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_test.ipynb
│   ├── 03_labeling_analysis.ipynb
│   └── 04_model_evaluation.ipynb
│
├── requirements.txt
├── Dockerfile
└── PIPELINE_DOCUMENTATION.md   ← questo file
```

---

### 0.4 File di Configurazione Base

**`configs/azure.yaml`**
```yaml
azure:
  resource_group: rg-glass-cleanliness
  storage_account: stglasscleanliness
  location: italynorth
  
  containers:
    raw_images: raw-images
    annotations: annotations
    processed: processed
    models: models
    outputs: outputs

  ml_workspace: aml-glass-cleanliness
  vm_training: vm-training-gpu
```

**`configs/model.yaml`**
```yaml
segmentation:
  model: grounding_dino + sam2
  dino_checkpoint: groundingdino_swint_ogc.pth
  sam2_checkpoint: sam2_hiera_large.pt
  confidence_threshold: 0.7
  text_prompts:
    - "glass window"
    - "glazed facade"
    - "curtain wall"

classification:
  model: efficientnet_b2
  num_classes: 4
  input_size: 224
  
classes:
  0: clean
  1: slightly_dirty
  2: moderately_dirty
  3: heavily_dirty
```

**`configs/training.yaml`**
```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adamw
  scheduler: cosine_annealing
  early_stopping_patience: 10
  
  augmentation:
    horizontal_flip: true
    random_brightness: 0.2
    random_contrast: 0.2
    gaussian_noise: true
    motion_blur: true   # simula movimento drone
    
  loss: focal_loss      # gestisce classi sbilanciate
  focal_gamma: 2.0
```

---

### 0.5 Checklist Fine Fase 0

- [ ] Resource Group Azure creato
- [ ] Storage Account + 5 container Blob creati
- [ ] Azure ML Workspace attivo
- [ ] VM GPU avviata e accessibile via SSH
- [ ] Docker + CVAT installati e raggiungibili su porta 8080
- [ ] Ambiente Conda `glass-cv` configurato con tutte le dipendenze
- [ ] Repository Git inizializzato con la struttura delle cartelle
- [ ] File di configurazione YAML compilati con i propri parametri

---

## FASE 1 — Raccolta e Preparazione Immagini

> *(Documentazione da completare nella prossima iterazione)*

---

## FASE 2 — Segmentazione Automatica delle Vetrate

> *(Documentazione da completare nella prossima iterazione)*

---

## FASE 3 — Human Refinement con CVAT

> *(Documentazione da completare nella prossima iterazione)*

---

## FASE 4 — AI-Assisted Labeling della Pulizia

> *(Documentazione da completare nella prossima iterazione)*

---

## FASE 5 — Training del Modello Finale

> *(Documentazione da completare nella prossima iterazione)*

---

## FASE 6 — Heatmap e Output Visivo

> *(Documentazione da completare nella prossima iterazione)*

---

## Stack Tecnologico

| Componente | Tecnologia | Motivo della scelta |
|---|---|---|
| Auto-rilevamento vetrate | Grounding DINO | Rileva oggetti tramite descrizione testuale, nessun training necessario |
| Segmentazione precisa | SAM2 (Meta) | Stato dell'arte nella segmentazione, zero-shot |
| Annotation tool | CVAT | Open source, supporta import/export COCO, web-based |
| AI Labeling assistito | CLIP (OpenAI) | Classifica immagini tramite testo, senza esempi etichettati |
| Training classificatore | PyTorch Lightning | Scalabile, integrazione nativa con MLflow |
| Backbone classificatore | EfficientNet-B2 | Ottimo rapporto accuratezza/peso, adatto a edge deployment |
| Heatmap | Grad-CAM | Mostra quali pixel influenzano la predizione |
| Experiment tracking | MLflow + Azure ML | Storico di tutti gli esperimenti, confronto metriche |
| Serving modello | Azure ML Endpoint | API REST per integrazione con sistemi BMS |
| Infrastruttura | Azure (A100 GPU) | GPU A100 ottimale per training rapido |

---

## Note Finali

- Questa documentazione viene aggiornata iterazione per iterazione
- Ogni fase produce **artifact verificabili** (file, metriche, checkpoint) prima di procedere alla fase successiva
- Il sistema è progettato per essere **incrementalmente migliorabile**: si può ri-addestrare il classificatore man mano che crescono i dati etichettati
- L'architettura a due stadi (segmentazione separata dalla classificazione) permette di aggiornare i due moduli indipendentemente
