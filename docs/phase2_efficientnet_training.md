# Fase 2 — Training EfficientNet Cleanliness Classifier
## Documentazione Tecnica e Funzionale

**Progetto:** Glass Cleanliness Detection — AI Computer Vision  
**Step:** 2 di 6 — Classificazione automatica della pulizia delle finestre  
**Data:** Aprile 2026  
**Stato:** ✅ Training completato — modello salvato in `models/weights/efficientnet_best.pt`  
**Prerequisito:** Fase 1 completata (YOLOv8 v3 operativo, mAP50=0.947)

---

## Indice

1. [Obiettivo](#1-obiettivo)
2. [Architettura del Modello](#2-architettura-del-modello)
3. [Costruzione del Dataset](#3-costruzione-del-dataset)
4. [Pipeline di Preparazione delle Immagini](#4-pipeline-di-preparazione-delle-immagini)
5. [Strategia di Training in Due Fasi](#5-strategia-di-training-in-due-fasi)
6. [Dettagli Tecnici del Training](#6-dettagli-tecnici-del-training)
7. [Risultati](#7-risultati)
8. [Script e File Coinvolti](#8-script-e-file-coinvolti)
9. [Considerazioni sulla Qualità del Dataset](#9-considerazioni-sulla-qualità-del-dataset)
10. [Prossimi Step](#10-prossimi-step)

---

## 1. Obiettivo

In questa fase addestriamo il secondo modello della pipeline: un classificatore che riceve in input il **crop di una singola finestra** (estratto da YOLOv8) e produce in output l'etichetta **clean** o **dirty** con un confidence score.

```
[YOLOv8] → bbox finestra → crop → [EfficientNet-B2] → clean / dirty
```

Il modello non vede l'intera facciata, ma solo la singola finestra già ritagliata. Questo è fondamentale: permette al classificatore di concentrarsi esclusivamente sulla texture e sull'aspetto del vetro, senza essere influenzato dallo sfondo dell'edificio.

---

## 2. Architettura del Modello

### EfficientNet-B2

Utilizziamo **EfficientNet-B2** dalla libreria `timm`, pretrained su ImageNet.

| Parametro | Valore |
|---|---|
| Architettura | EfficientNet-B2 |
| Input size | 260×260 px (nativo B2) |
| Numero classi | 2 (clean=0, dirty=1) |
| Parametri totali | ~9.2M |
| Libreria | timm 1.0.26 |

### Perché EfficientNet-B2 e non altre architetture?

| Architettura | Pro | Contro |
|---|---|---|
| **EfficientNet-B2** | Ottimo rapporto accuracy/peso, adatto a edge deployment | — |
| ResNet-50 | Semplice e collaudata | Più pesante, meno efficiente |
| ViT (Vision Transformer) | SOTA su grandi dataset | Richiede molti dati, lento su GPU piccole |
| MobileNet | Leggerissimo | Accuracy inferiore su texture sottili |

EfficientNet è progettato per essere efficiente: scala la profondità, la larghezza e la risoluzione della rete in modo bilanciato. Per classificare la pulizia di superfici vetrate — un task che richiede riconoscimento di texture sottili (polvere, aloni, macchie) — questo equilibrio è ideale.

---

## 3. Costruzione del Dataset

### 3.1 Fonti delle Immagini

Il dataset è costruito a partire da **tre fonti distinte**, ognuna corrispondente a una sottocartella in `data/windows/train/clean/` e `data/windows/train/dirty/`:

| Sottocartella | Fonte | Tipo | Note |
|---|---|---|---|
| `CLEN_CROPPED_AMI` / `DIRTY_CROPPED_AMI` | Immagini scattate manualmente | Fotografie reali | Massima variabilità, dominio target |
| `Finestre_pulite_CROPPED_Giuse` / `Finestre_sporche_CROPPED_Giuse` | Immagini da Pexels (API) | Fotografie stock | Buona qualità, vario contesto urbano |
| `clean_crop_pexels` / `dirty_crop_pexels` | Immagini generate/scaricate via script | Misto | Integrazione automatica |

**Totale immagini al momento del training:**

| Classe | Immagini |
|---|---|
| clean | 919 |
| dirty | 1.004 |
| **Totale** | **1.923** |

Il dataset è **sostanzialmente bilanciato** (≈48% clean, ≈52% dirty), il che riduce il rischio di bias verso una delle due classi.

### 3.2 Come sono state generate le immagini

Tutte le immagini presenti nel dataset sono **crop di finestre reali o fotografate**, non immagini di facciate intere. Il processo di generazione è il seguente:

1. Si parte da immagini sorgente (fotografie di edifici/facciate)
2. Il modello YOLOv8 v3 (già addestrato nella Fase 1) viene eseguito su ogni immagine
3. Ogni finestra rilevata con confidenza ≥ 60% e dimensione ≥ 66×66 px viene ritagliata
4. Viene aggiunto un padding del 5% attorno al bounding box per non tagliare i bordi della finestra
5. Il crop viene salvato come file `.jpg` e poi etichettato manualmente come clean o dirty

Questo approccio garantisce che il classificatore veda esattamente lo stesso tipo di input che riceverà durante l'inferenza reale.

### 3.3 Split Stratificato

Lo script `src/data/build_efficientnet_dataset.py` divide le immagini in **train / valid / test** con proporzione **70% / 15% / 15%**.

Lo split è **stratificato per fonte**: ogni sottocartella (es. `DIRTY_CROPPED_AMI`) contribuisce in egual misura a tutti e tre gli split. Questo evita che una fonte finisca interamente in train e un'altra interamente in test, il che falsificherebbe la valutazione.

```
data/windows_efficientnet/
├── train/
│   ├── clean/   (≈643 immagini)
│   └── dirty/   (≈703 immagini)
├── valid/
│   ├── clean/   (≈138 immagini)
│   └── dirty/   (≈150 immagini)
└── test/
    ├── clean/   (≈138 immagini)
    └── dirty/   (≈151 immagini)
```

Il manifest CSV (`manifest.csv`) registra per ogni immagine: filename, label, sottocartella sorgente, split di appartenenza e path originale — così la provenienza non viene mai persa.

---

## 4. Pipeline di Preparazione delle Immagini

### 4.1 Flusso Completo

```
Immagini sorgente (facciate edifici, fotografie)
         │
         ▼
[filter_and_crop_windows.py]
  YOLOv8 v3 (conf ≥ 0.60, size ≥ 66px)
  → crop con padding 5%
  → salva in data/windows/train/clean/ o dirty/
         │
         ▼
[build_efficientnet_dataset.py]
  Split stratificato 70/15/15
  Prefissa filename con nome fonte
  Genera manifest.csv
         │
         ▼
data/windows_efficientnet/train|valid|test/clean|dirty/
         │
         ▼
[train_efficientnet.py]
  Fase 1 — warmup (backbone frozen)
  Fase 2 — fine-tuning (tutti i layer)
         │
         ▼
models/weights/efficientnet_best.pt
```

### 4.2 Data Augmentation (solo su train)

Durante il training viene applicata augmentation per aumentare la variabilità artificiale e rendere il modello robusto a condizioni reali:

| Trasformazione | Parametro | Motivazione |
|---|---|---|
| Resize + RandomCrop | 280→260 px | Variazione lieve di framing |
| RandomHorizontalFlip | p=0.5 | Simmetria sinistra/destra |
| RandomVerticalFlip | p=0.2 | Angolazioni drone diverse |
| ColorJitter | brightness±0.3, contrast±0.3, saturation±0.2 | Ore del giorno, condizioni meteo |
| RandomRotation | ±10° | Leggera inclinazione del drone |
| Normalize | mean=[0.485,0.456,0.406] | Standard ImageNet pretrained |

Su valid e test si usa solo Resize(260) + Normalize, senza augmentation, per valutare il modello in condizioni stabili.

---

## 5. Strategia di Training in Due Fasi

Questa è la parte più importante da capire per interpretare i log di training.

### 5.1 Cos'è il Transfer Learning

EfficientNet-B2 viene scaricato con i **pesi ImageNet**: parametri già ottimizzati su 1.2 milioni di immagini di 1000 classi (cani, automobili, strumenti musicali...). Questi pesi non servono direttamente per il nostro task, ma il backbone ha già imparato a riconoscere **bordi, texture, pattern di colore e forme geometriche** — capacità che sono utili anche per rilevare sporco su vetro.

Il transfer learning sfrutta questa conoscenza pre-acquisita invece di ricominciare da zero.

```
ImageNet (1.2M immagini, 1000 classi)
         │ training da Torchvision/timm
         ▼
EfficientNet-B2 pretrained  ← pesi già utili per texture
         │ fine-tuning (il nostro training)
         ▼
EfficientNet-B2 specializzato clean/dirty
```

### 5.2 Fase 1 — Warmup (backbone frozen, 5 epoche)

Nel warmup il backbone (tutti i layer che estraggono feature dalle immagini) viene **congelato**: i suoi pesi non cambiano. Viene aggiornato **solo il classifier head**, cioè gli ultimi layer lineari che producono le probabilità per le 2 classi.

**Perché?**  
Se sbloccassimo subito tutto il modello con learning rate alto, i gradienti calcolati dal classifier head (inizializzato casualmente) potrebbero "rovinare" i pesi ImageNet già buoni nel backbone, soprattutto nelle prime epoch quando il head non sa ancora cosa fare.

Il warmup "scalda" il head prima di procedere: dopo 5 epoch il classifier sa già orientarsi tra clean e dirty, anche se con performance modeste.

**Cosa si vede nei log:**
```
[warmup] epoch 01/5  train_acc=0.446  val_f1_dirty=0.497   ← quasi random (50%)
[warmup] epoch 05/5  train_acc=0.493  val_f1_dirty=0.554   ← lieve miglioramento
```
Il warmup produce risultati modesti (~55% F1) perché il backbone è bloccato e non può adattarsi al dominio. È atteso.

### 5.3 Fase 2 — Fine-tuning (tutti i layer sbloccati, 30 epoche)

Nel fine-tuning **tutti i parametri del modello vengono sbloccati** e aggiornati, con un learning rate 10 volte inferiore rispetto al warmup (per non distruggere i pesi ImageNet con aggiornamenti troppo bruschi).

Il backbone può ora adattarsi al dominio specifico delle finestre: impara a riconoscere le texture di polvere, aloni, macchie su vetro invece di classificare oggetti generici.

**Cosa si vede nei log:**
```
[finetune] epoch 01/30  train_acc=0.746  val_f1_dirty=0.816   ← salto enorme
[finetune] epoch 02/30  train_acc=0.897  val_f1_dirty=0.871
[finetune] epoch 03/30  train_acc=0.930  val_f1_dirty=0.906
[finetune] epoch 04/30  train_acc=0.941  val_f1_dirty=0.917
[finetune] epoch 06/30  train_acc=0.961  val_f1_dirty=0.931
```

Il salto drammatico dalla epoch 5 warmup alla epoch 1 finetune (da 0.55 → 0.82 F1) è normale e atteso: nel momento in cui il backbone viene sbloccato, la rete può finalmente adattarsi al dominio.

### 5.4 Perché questo approccio è migliore del fine-tuning diretto

| Approccio | Rischio | Risultato |
|---|---|---|
| Fine-tuning diretto (senza warmup) | Gradienti instabili nelle prime epoch → pesi ImageNet danneggiati | Convergenza più lenta, performance inferiori |
| **Warmup + fine-tuning (nostro)** | Nessuno | Convergenza rapida, sfrutta al massimo ImageNet |
| Training from scratch | Richiede molti più dati e tempo | Non praticabile con ~1900 immagini |

---

## 6. Dettagli Tecnici del Training

### Configurazione

| Parametro | Valore | Motivazione |
|---|---|---|
| Optimizer | AdamW | Variante di Adam con weight decay, standard per fine-tuning |
| LR warmup | 3×10⁻⁴ | Solo il head viene aggiornato |
| LR finetune | 3×10⁻⁵ | 10× più basso per non danneggiare backbone |
| Scheduler | CosineAnnealingLR | Riduce LR gradualmente fino a LR/100 — evita oscillazioni |
| Weight decay | 1×10⁻⁴ | Regularizzazione contro overfitting |
| Batch size | 32 | Ottimale per GPU A10-8Q (8GB VRAM) |
| Loss | CrossEntropyLoss (con class weights) | Gestisce eventuale sbilanciamento tra classi |

### Class Weights

Per gestire lo sbilanciamento tra classi (anche lieve), la loss è pesata inversamente alla frequenza di ogni classe:

```python
weights = 1.0 / counts_per_class
weights = weights / weights.sum() * num_classes
```

Questo fa sì che un errore su una classe rara pesi di più nella loss rispetto a una classe frequente.

### Metrica di Selezione del Best Model

Il modello migliore viene salvato in base a **val F1-dirty** (F1 score sulla classe dirty), non sull'accuracy generale.

**Perché F1-dirty e non accuracy?**  
Nel nostro dominio, **classificare erroneamente una finestra sporca come pulita è più grave** del contrario (genera falsi negativi: finestre sporche non segnalate). L'F1 score è la media armonica di precision e recall — bilancia i due tipi di errore. Ottimizzare su F1-dirty massimizza la capacità del modello di trovare le finestre sporche.

### Infrastruttura VM

- **GPU:** NVIDIA A10-8Q (8GB VRAM, CUDA 12.8)
- **Conda env:** `glass-cv` (Python 3.10, PyTorch 2.5.1+cu121, timm 1.0.26)
- **Durata training:** ~5 min warmup + ~10 min finetune = ~15 min totali
- **Tracking:** MLflow experiment `efficientnet_cleanliness` su `http://20.9.194.236:5000`

---

## 7. Risultati

### Curve di Training

| Fase | Epoch | train_loss | train_acc | val_loss | val_acc | val_f1_dirty |
|---|---|---|---|---|---|---|
| warmup | 1 | 3.110 | 0.446 | 2.691 | 0.482 | 0.497 |
| warmup | 3 | 2.566 | 0.472 | 2.440 | 0.504 | 0.537 |
| warmup | 5 | 2.583 | 0.493 | 2.296 | 0.525 | 0.554 |
| finetune | 1 | 0.943 | 0.746 | 0.680 | 0.820 | 0.816 |
| finetune | 2 | 0.375 | 0.897 | 0.496 | 0.871 | 0.871 |
| finetune | 3 | 0.251 | 0.930 | 0.323 | 0.903 | 0.906 |
| finetune | 4 | 0.207 | 0.941 | 0.321 | 0.914 | 0.917 |
| finetune | 6 | 0.154 | 0.961 | 0.268 | 0.928 | 0.931 |

### Interpretazione

- **Warmup (epoch 1-5):** progressione lenta ma stabile. Il backbone bloccato limita la capacità del modello.
- **Finetune epoch 1:** il salto da 0.55 → 0.82 conferma che il backbone si specializza rapidamente nel riconoscere texture di sporco/pulizia su vetro, sfruttando la conoscenza ImageNet come punto di partenza.
- **Finetune epoch 6+:** convergenza verso 0.93+ F1-dirty. La val_loss scende insieme alla train_loss → nessun overfitting significativo.

### Valutazione Finale (test set)

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| clean | 0.948 | 0.984 | 0.966 | 129 |
| dirty | 0.986 | 0.953 | 0.969 | 149 |
| **accuracy** | | | **0.968** | **278** |
| macro avg | 0.967 | 0.969 | 0.968 | 278 |
| weighted avg | 0.968 | 0.968 | 0.968 | 278 |

**Test accuracy: 96.8% — F1-dirty: 96.9%** ✅ Obiettivo superato (target era >85% acc, >80% F1-dirty).

**Interpretazione:**
- **Precision dirty = 0.986**: quando il modello dice "sporca", ha ragione nel 98.6% dei casi → quasi nessun falso allarme
- **Recall dirty = 0.953**: trova il 95.3% di tutte le finestre sporche → pochissime finestre sporche silenziosamente classificate come pulite
- Il modello è più conservativo sul clean (recall=0.984) che sulla detection dello sporco (recall=0.953), che è il comportamento desiderato: meglio segnalare un'eventuale pulizia in più che perderne una necessaria

---

## 8. Script e File Coinvolti

| File | Ruolo |
|---|---|
| `src/data/filter_and_crop_windows.py` | Usa YOLOv8 per croppare finestre da immagini sorgente |
| `src/data/build_efficientnet_dataset.py` | Costruisce train/valid/test con split stratificato |
| `src/training/train_efficientnet.py` | Training EfficientNet-B2 (warmup + fine-tuning) |
| `src/inference/analyze_facade.py` | Pipeline end-to-end: YOLOv8 → EfficientNet → Grad-CAM → report JSON |
| `data/windows/train/clean/` | Immagini sorgente etichettate clean (3 sottocartelle) |
| `data/windows/train/dirty/` | Immagini sorgente etichettate dirty (3 sottocartelle) |
| `data/windows_efficientnet/` | Dataset pronto per training (generato, non in git) |
| `data/windows_efficientnet/manifest.csv` | Tracciabilità per ogni immagine: fonte, label, split |
| `models/weights/efficientnet_best.pt` | Modello migliore (non in git, su Blob Storage) |

---

## 8b. Pipeline di Inferenza End-to-End (`analyze_facade.py`)

Lo script `src/inference/analyze_facade.py` esegue l'intera pipeline su una singola immagine di facciata:

```
Immagine facciata
      │
      ▼
[YOLOv8 v3]  conf ≥ 0.60, size ≥ 66px
      │  bounding box finestre
      ▼
[filter_contained_boxes]  ← rimuove box annidati
      │  solo finestre "foglia" (non contenitori)
      ▼
[EfficientNet-B2]  crop + padding 5%
      │  clean / dirty + confidence
      ▼
[Grad-CAM]  solo per finestre dirty
      │  heatmap dove si concentra lo sporco
      ▼
Immagine annotata + report JSON
```

**Uso:**
```bash
python src/inference/analyze_facade.py --image am_images_test/AMI1.png
# Output: runs/inference/AMI1_analyzed.jpg + AMI1_report.json
```

**Output immagine annotata:**
- Box **verde** = finestra clean, con label `clean XX%`
- Box **rosso** = finestra dirty, con label `dirty XX%`
- **Heatmap** colorata (rosso=sporco, blu=pulito) sovrapposta alle sole finestre dirty

**Output JSON (esempio):**
```json
{
  "image": "AMI1.png",
  "total_windows": 12,
  "clean": 8,
  "dirty": 4,
  "windows": [
    {
      "window_id": 0,
      "bbox_yolo": [120, 80, 320, 260],
      "label": "dirty",
      "conf_classifier": 0.9821,
      "dirty_prob": 0.9821,
      "clean_prob": 0.0179
    }
  ]
}
```

### Filtro Box Annidati

YOLOv8 a volte rileva sia una finestra "contenitore" (es. un serramento grande) che le singole finestre interne. La NMS standard non elimina questo caso perché i box non si sovrappongono sufficientemente — si *contengono*.

La funzione `filter_contained_boxes` risolve il problema: per ogni coppia di box (A, B), se l'80% dell'area di A è coperta da B e B è più grande → B viene scartato. Vengono mantenute le finestre più piccole (che rappresentano il vetro reale) e rimossa quella grande che le contiene.

```
Prima:  [finestra grande] + [finestra piccola 1] + [finestra piccola 2]
Dopo:                       [finestra piccola 1] + [finestra piccola 2]
```

La soglia di contenimento è configurabile via `--contain-thresh` (default: 0.80).

---

## 9. Considerazioni sulla Qualità del Dataset

### Punti di forza
- **Tre fonti diverse** (manuale, Pexels, AI) introducono variabilità di dominio: stili fotografici diversi, condizioni di luce diverse, qualità diverse.
- **Crop prodotti dallo stesso modello YOLOv8** usato in inferenza → distribuzione train/test coerente.
- **Split stratificato per fonte** → nessun leak di dominio tra train e test.

### Limiti noti
- I crop prodotti da YOLOv8 non sono sempre perfettamente centrati sulla finestra: alcuni includono cornici, muri o parti di edificio. Questo introduce **rumore sistematico**, ma agisce anche come augmentation implicita e rende il modello più robusto.
- Le immagini Pexels "clean" tendono a essere più "da studio" (luce perfetta, contrasto elevato) rispetto alle fotografie manuali → leggero distributional shift tra le due fonti.
- Con ~1900 immagini il dataset è **piccolo**: il transfer learning da ImageNet è essenziale per compensare.

### Miglioramenti futuri
1. Aggiungere le ~250 immagini in riserva dopo aver valutato le performance iniziali
2. Raccogliere più immagini di finestre con sporco borderline (leggermente sporche) per rendere il confine decisionale più robusto
3. Aggiungere una terza classe `slightly_dirty` se il cliente necessita di una prioritizzazione più fine degli interventi

---

## 10. Prossimi Step

1. **Scaricare `efficientnet_best.pt` da VM → Blob Storage** con:
   ```bash
   az storage blob upload \
     --account-name stglasscleanliness \
     --account-key "$key" \
     --container-name models \
     --name efficientnet_best.pt \
     --file /mnt/project/AI-CHALLENGE-AM-BOLO/models/weights/efficientnet_best.pt \
     --overwrite
   ```

2. **Creare `src/inference/analyze_facade.py`** — pipeline end-to-end:
   - YOLOv8 → detect finestre
   - Crop di ogni finestra
   - EfficientNet → clean/dirty + confidence
   - Grad-CAM → heatmap dello sporco
   - Report JSON con coordinate + label + score per ogni finestra

3. **Grad-CAM** — si ottiene gratuitamente dal modello già trainato:
   ```python
   from pytorch_grad_cam import GradCAM
   cam = GradCAM(model, target_layers=[model.blocks[-1]])
   heatmap = cam(input_tensor)
   dirty_score = heatmap.mean()  # % superficie sporca stimata
   ```
