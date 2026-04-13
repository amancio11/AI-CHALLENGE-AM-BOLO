# Fase 0 — Setup Infrastruttura Azure
## Documentazione Tecnica e Funzionale

**Progetto:** Glass Cleanliness Detection — AI Computer Vision  
**Step:** 0 di 6 — Provisioning dell'infrastruttura cloud  
**Data esecuzione:** Aprile 2026  
**Ambiente locale:** Windows 11, Git Bash (MINGW64) / SSH, Azure CLI  
**Stato:** ✅ VM operativa con GPU — ambiente conda installato

---

## Indice

1. [Obiettivo Funzionale](#1-obiettivo-funzionale)
2. [Architettura delle Risorse Azure](#2-architettura-delle-risorse-azure)
3. [Prerequisiti](#3-prerequisiti)
4. [Risorse Azure Esistenti e Riutilizzate](#4-risorse-azure-esistenti-e-riutilizzate)
5. [Provisioning — Script Automatico](#5-provisioning--script-automatico)
6. [Setup Ambiente VM](#6-setup-ambiente-vm)
7. [Struttura Blob Storage](#7-struttura-blob-storage)
8. [Trasferimento Dati con AzCopy](#8-trasferimento-dati-con-azcopy)
9. [Variabili d'Ambiente](#9-variabili-dambiente)
10. [Verifica del Setup](#10-verifica-del-setup)
11. [Stato Attuale VM (Aprile 2026)](#11-stato-attuale-vm-aprile-2026)
12. [Problemi Noti e Soluzioni](#12-problemi-noti-e-soluzioni)
13. [Prossimi Step](#13-prossimi-step)

---

## 1. Obiettivo Funzionale

La Fase 0 predispone l'intera infrastruttura cloud necessaria per le fasi successive del progetto. Si esegue **una sola volta** all'inizio del progetto.

Al termine di questa fase saranno operativi:

| Componente | URL / Indirizzo | Scopo |
|---|---|---|
| **Azure Blob Storage** | `stglasscleanliness.blob.core.windows.net` | Centralizzare immagini raw, annotazioni, modelli e output |
| **CVAT** (annotazione) | `http://20.9.194.236:8080` | Annotare manualmente le vetrate e i livelli di pulizia |
| **MLflow** (tracking) | `http://20.9.194.236:5000` | Tracciare esperimenti, metriche e artefatti del training |
| **conda env `glass-cv`** | sulla VM, via SSH o run-command | Eseguire i training PyTorch con supporto CUDA |

---

## 2. Architettura delle Risorse Azure

### Scelta Architetturale: VM All-in-One

Per semplicità e ottimizzazione dei costi, **tutto è ospitato su una singola VM GPU**:

```
┌─────────────────────────────────────────────────────────────┐
│  BOLOGNA-AI-AM-MACHINE  (Standard_NV12ads_A10_v5)           │
│  IP pubblico: 20.9.194.236   RG: kpmg-bologna               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  CVAT        │  │  MLflow      │  │  Conda env       │  │
│  │  porta 8080  │  │  porta 5000  │  │  glass-cv        │  │
│  │  (Docker)    │  │  (systemd)   │  │  PyTorch+CUDA    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
             │
             │ accesso sicuro
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Azure Blob Storage: stglasscleanliness                     │
│  RG: kpmg-bologna                                           │
│  Container: raw-images | annotations | models | results     │
└─────────────────────────────────────────────────────────────┘
```

Questa scelta **non** utilizza:
- Azure ML Workspace (non necessario per il prototipo)
- Azure Container Instances separate
- Più VM con Load Balancer

### Specifiche VM

| Parametro | Valore |
|---|---|
| Nome | `BOLOGNA-AI-AM-MACHINE` |
| Resource Group | `kpmg-bologna` |
| Location | `westus2` |
| SKU | `Standard_NV12ads_A10_v5` |
| GPU | NVIDIA A10-8Q (8 GB VRAM, CUDA 12.8) |
| Driver NVIDIA | 570.211.01 (installato via Azure Extension `NvidiaGpuDriverLinux`) |
| OS | Ubuntu 24.04 LTS (kernel 6.17.0-1010-azure) |
| IP pubblico | `20.9.194.236` |
| Autenticazione | SSH con chiave `.pem` (funzionante) |
| Username OS | `azureuseram` |

> **Nota SSH:** La VM è accessibile via `ssh azureuseram@20.9.194.236` dalla rete locale. Per accesso da altri ambienti usare `az ssh vm` o il portale Azure Serial Console.

---

## 3. Prerequisiti

### Lato Locale (Windows)

- **Azure CLI** installato e in PATH (`az --version`)
- **Git Bash (MINGW64)** oppure PowerShell
- Account Azure con ruolo `Contributor` sul Resource Group `kpmg-bologna`
- Python 3.10+ (opzionale, per script locali)

### Login Azure

```bash
az login
# Segui il browser per autenticarsi
az account show  # verifica account attivo
```

---

## 4. Risorse Azure Esistenti e Riutilizzate

> Questo progetto è stato sviluppato su infrastruttura **già esistente** anziché crearne di nuova da zero.

### Resource Group

```
Nome:     kpmg-bologna
Location: westus2
Contiene: VM + Storage Account (entrambi nello stesso RG)
```

Il Resource Group è preesistente. Lo script `provision_azure.sh` **verifica** che esista ma **non lo crea**.

### Storage Account

```
Nome:     stglasscleanliness
RG:       kpmg-bologna        ← spostato qui manualmente dal RG originale
SKU:      Standard_LRS
```

Lo script crea i container Blob se non esistono già, senza toccare quelli esistenti.

> **Nota:** Esiste anche lo storage account `kpmgbolognaai` nello stesso RG — verificare quale dei due usare per i trasferimenti (entrambi funzionanti).

### VM

La VM `BOLOGNA-AI-AM-MACHINE` è preesistente. Lo script:
1. Verifica che sia presente nel RG
2. Apre le porte NSG necessarie (8080 e 5000) se non già aperte
3. Trasferisce ed esegue `setup_vm.sh` sull'ambiente VM per completare il setup software

---

## 5. Provisioning — Script Automatico

### File: `scripts/provision_azure.sh`

**Esecuzione** (dalla directory `scripts/`):

```bash
cd scripts/
bash provision_azure.sh
```

### Cosa fa lo script (step by step)

#### Step 0 — Verifica login
```bash
az account show
```
Se non autenticato, avvia `az login` automaticamente.

#### Step 1 — Verifica Resource Groups
Controlla che `kpmg-bologna` esista. Se non trovato, esce con errore (non crea RG per sicurezza).

#### Step 2 — Storage Account e Container Blob
```bash
# Verifica lo storage account
az storage account show --name stglasscleanliness

# Crea i 4 container (idempotente: skip se già esistenti)
az storage container create --name raw-images   --auth-mode key
az storage container create --name annotations  --auth-mode key
az storage container create --name models       --auth-mode key
az storage container create --name results      --auth-mode key

# Salva la connection string in .env.local
az storage account show-connection-string ... > .env.local
```

> **Nota tecnica:** tutte le operazioni storage usano `--auth-mode key` per evitare problemi con i permessi RBAC dell'account Azure AD.

#### Step 3 — VM: porte NSG
```bash
az vm open-port --port 8080 --priority 1001   # CVAT
az vm open-port --port 5000 --priority 1002   # MLflow
```
Idempotente: se la porta è già aperta, il comando vine ignorato silenziosamente.

#### Step 4 — Trasferimento setup_vm.sh sulla VM

Poiché la VM usa AAD auth (no SSH diretto), e `az vm run-command` su Git Bash Windows ha limitazioni di quoting, il trasferimento usa questa strategia:

```bash
# 1. Upload dello script su Blob Storage
az storage blob upload \
  --container-name models \
  --name scripts/setup_vm.sh \
  --file setup_vm.sh \
  --auth-mode key --overwrite

# 2. Genera SAS URL temporaneo (2 ore, read-only)
SAS=$(az storage blob generate-sas \
  --account-name stglasscleanliness \
  --container-name models \
  --name scripts/setup_vm.sh \
  --permissions r \
  --expiry 2026-04-13T23:59:00Z \
  --auth-mode key \
  --output tsv)

BLOB_URL="https://stglasscleanliness.blob.core.windows.net/models/scripts/setup_vm.sh?${SAS}"

# 3. Base64-encode dell'URL (elimina & % ? dai problemi di quoting)
B64=$(echo -n "$BLOB_URL" | base64 | tr -d '\n')

# 4. wget sulla VM passando l'URL come base64 inline
az vm run-command invoke \
  -g kpmg-bologna -n BOLOGNA-AI-AM-MACHINE \
  --command-id RunShellScript \
  --scripts "echo ${B64} | base64 -d | wget -q -i - -O /tmp/setup_vm.sh && bash /tmp/setup_vm.sh > /tmp/setup.log 2>&1 </dev/null & disown; echo SETUP_STARTED"
```

**Perché base64?** L'URL SAS contiene caratteri `&`, `%`, `=` e `?` che vengono mal interpretati da `az` CLI su Git Bash (MSYS path conversion). Codificando in base64 si ottiene una stringa `[A-Za-z0-9+/=]` che az passa integra all'API ARM.

---

## 6. Setup Ambiente VM

### Cosa installa (in ordine)

| # | Componente | Stato attuale |
|---|---|---|
| 1 | **Docker** + Docker Compose | ✅ Installato (v29.4.0) |
| 2 | **Miniconda** su `/mnt/miniconda` | ✅ Installato (conda 26.1.1) |
| 3 | **conda env `glass-cv`** — Python 3.10, PyTorch 2.5.1+cu121 | ✅ Installato |
| 4 | **ultralytics** (YOLOv8), timm, mlflow, opencv, scikit-learn | ✅ Installato |
| 5 | **Driver NVIDIA** A10-8Q | ✅ Driver 570 via Azure Extension |
| 6 | **CVAT** | ⏳ Sorgente presente in `/mnt/cvat`, da avviare |
| 7 | **MLflow server** | ⏳ Da avviare manualmente |

> **RIMOSSO dalla pipeline:** SAM2 weights (~900MB) e Grounding DINO weights (~700MB) — non più necessari nella nuova architettura YOLOv8 + EfficientNet.

### Monitoraggio installazione

```bash
# Controlla il log in tempo reale (aggiorna ogni ~60 secondi manualmente)
az vm run-command invoke \
  -g kpmg-bologna -n BOLOGNA-AI-AM-MACHINE \
  --command-id RunShellScript \
  --scripts "tail -50 /tmp/setup.log"
```

### Tempo atteso

| Fase | Tempo stimato |
|---|---|
| Docker + Conda | ~5 minuti |
| Pytorch + dipendenze | ~10 minuti |
| CVAT (build Docker) | ~15 minuti |
| Download pesi modelli | ~10 minuti (dipende dalla banda) |
| **Totale** | **~30-40 minuti** |

---

## 7. Struttura Blob Storage

Lo storage account di riferimento è `stglasscleanliness` (RG `kpmg-bologna`).

La struttura rispecchia il flusso della pipeline: ogni fase legge e scrive in container separati.

```
stglasscleanliness/
│
├── datasets/                             ← INPUT dati grezzi e dataset etichettati
│   │
│   ├── window-detection-vnpow-v1/        ← Fase 1: dataset Roboflow COCO per YOLO training
│   │   ├── train/                        │   (immagini facciate + bounding box finestre)
│   │   ├── valid/
│   │   └── test/
│   │
│   ├── facades/                          ← Fase 2: immagini grezze di facciate da processare
│   │                                     │   (input per l'inferenza YOLO)
│   │
│   └── windows-labeled/                  ← Fase 3: finestre ritagliate + etichettate
│       ├── train/                        │   (output YOLO crop + labeling semi-manuale CVAT)
│       │   ├── clean/
│       │   └── dirty/
│       ├── valid/
│       │   ├── clean/
│       │   └── dirty/
│       └── test/
│           ├── clean/
│           └── dirty/
│
├── models/                               ← OUTPUT modelli trainati
│   ├── scripts/                          ← setup_vm.sh, scripts di supporto
│   ├── yolov8_windows_best.pt            ← output Fase 1 (YOLO window detector)
│   └── efficientnet_cleanliness.pt       ← output Fase 3 (classificatore clean/dirty)
│
├── annotations/                          ← LABEL export da CVAT
│   └── cleanliness/                      ← JSON con etichette clean/dirty per ogni finestra
│
└── results/                              ← OUTPUT inferenza end-to-end
    ├── detections/                       ← bounding box JSON da YOLO su nuove facciate
    └── reports/                          ← report finale pulizia per facciata (per KPMG)
```

### Corrispondenza locale ↔ Storage

| Cartella locale | Container Blob | Fase |
|---|---|---|
| `data/raw/roboflow/window-detection-vnpow-v1/` | `datasets/window-detection-vnpow-v1/` | Fase 1 |
| `data/facades/` | `datasets/facades/` | Fase 2 |
| `data/windows/{train,valid,test}/{clean,dirty}/` | `datasets/windows-labeled/` | Fase 3 |
| `models/weights/*.pt` | `models/` | Fase 1 e 3 |
| `data/annotations/cleanliness/` | `annotations/cleanliness/` | Fase 3 |

---

## 8. Trasferimento Dati con AzCopy

AzCopy v10 è lo strumento principale per muovere dati tra PC locale, VM e Storage.

> **Metodo di autenticazione:** si usa un **SAS token generato con account-key** (non Azure AD).
> Questo evita il problema RBAC (`AuthorizationPermissionMismatch`) e funziona sempre.

### Installazione AzCopy su Windows (Git Bash)

```bash
# Scarica il binario Windows (eseguire in Git Bash)
curl -sL https://aka.ms/downloadazcopy-v10-windows -o /tmp/azcopy.zip
cd /tmp && unzip -o azcopy.zip
mkdir -p "$HOME/bin"
mv azcopy_windows_amd64_*/azcopy.exe "$HOME/bin/azcopy.exe"
export PATH="$HOME/bin:$PATH"
"$USERPROFILE/bin/azcopy.exe" --version
# → azcopy version 10.x.x
```

> **Nota:** non usare il binario Linux (`downloadazcopy-v10-linux`) su Windows — non è eseguibile.
> Il PATH `$HOME/bin` non persiste tra sessioni: ri-esportarlo oppure aggiungerlo in `~/.bashrc`.

### Installazione AzCopy sulla VM Linux

```bash
# Sulla VM SSH (Ubuntu)
wget https://aka.ms/downloadazcopy-v10-linux -O /tmp/azcopy.tar.gz
tar -xf /tmp/azcopy.tar.gz -C /tmp/
sudo mv /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
azcopy --version
```

---

### Procedura Standard: Upload/Download con SAS token

**Ogni operazione azcopy segue questo schema in 3 passi:**

```bash
# PASSO 1 — ottieni la storage key
key=$(az storage account keys list \
  --account-name stglasscleanliness \
  --resource-group kpmg-bologna \
  --query "[0].value" -o tsv)

# PASSO 2 — crea il container se non esiste (idempotente)
az storage container create \
  --account-name stglasscleanliness \
  --name <CONTAINER_NAME> \
  --account-key "$key"

# PASSO 3 — genera SAS token per il container
sas=$(az storage container generate-sas \
  --account-name stglasscleanliness \
  --name <CONTAINER_NAME> \
  --permissions rwdlac \
  --expiry 2026-12-31T23:59:00Z \
  --account-key "$key" \
  --output tsv)

# PASSO 4 — esegui azcopy con SAS nell'URL
"$USERPROFILE/bin/azcopy.exe" copy \
  "<SORGENTE>" \
  "https://stglasscleanliness.blob.core.windows.net/<CONTAINER_NAME>?${sas}" \
  --recursive
```

---

### Upload dataset da PC locale → Storage

```bash
key=$(az storage account keys list --account-name stglasscleanliness --resource-group kpmg-bologna --query "[0].value" -o tsv) \
&& az storage container create --account-name stglasscleanliness --name datasets --account-key "$key" \
&& sas=$(az storage container generate-sas \
  --account-name stglasscleanliness --name datasets \
  --permissions rwdlac --expiry 2026-12-31T23:59:00Z \
  --account-key "$key" --output tsv) \
&& "$USERPROFILE/bin/azcopy.exe" copy \
  "C:/Users/andreamancini/Downloads/AI-CHALLENGE2/data/raw/roboflow" \
  "https://stglasscleanliness.blob.core.windows.net/datasets?${sas}" \
  --recursive
```

### Download Storage → VM

```bash
# Sulla VM SSH
key=$(az storage account keys list --account-name stglasscleanliness --resource-group kpmg-bologna --query "[0].value" -o tsv) \
&& sas=$(az storage container generate-sas \
  --account-name stglasscleanliness --name datasets \
  --permissions rl --expiry 2026-12-31T23:59:00Z \
  --account-key "$key" --output tsv) \
&& mkdir -p /mnt/project/data/raw/roboflow \
&& azcopy copy \
  "https://stglasscleanliness.blob.core.windows.net/datasets/window-detection-vnpow?${sas}" \
  "/mnt/project/data/raw/roboflow/" --recursive
```

### Upload modello trainato → Storage

```bash
key=$(az storage account keys list --account-name stglasscleanliness --resource-group kpmg-bologna --query "[0].value" -o tsv) \
&& az storage container create --account-name stglasscleanliness --name models --account-key "$key" \
&& sas=$(az storage container generate-sas \
  --account-name stglasscleanliness --name models \
  --permissions rwdlac --expiry 2026-12-31T23:59:00Z \
  --account-key "$key" --output tsv) \
&& azcopy copy \
  "/mnt/project/runs/window_detection/yolov8n_windows/weights/best.pt" \
  "https://stglasscleanliness.blob.core.windows.net/models/yolov8_windows_best.pt?${sas}"
```

---

## 9. Variabili d'Ambiente

### File: `.env.local` (generato automaticamente dallo script)

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=stglasscleanliness;...
VM_PUBLIC_IP=20.9.194.236
```

> `.env.local` è escluso da git (nel `.gitignore`). Non committare mai credenziali.

### File: `.env` (da creare manualmente)

```bash
cp .env.local .env
echo "ROBOFLOW_API_KEY=<la_tua_api_key>" >> .env
echo "MLFLOW_TRACKING_URI=http://20.9.194.236:5000" >> .env
```

### Template: `.env.example` (versionato in git)

```env
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
VM_PUBLIC_IP=20.9.194.236
ROBOFLOW_API_KEY=your_roboflow_key_here
MLFLOW_TRACKING_URI=http://20.9.194.236:5000
```

---

## 10. Verifica del Setup

### 10.1 GPU e CUDA

```bash
# Dalla SSH sulla VM
nvidia-smi
# Atteso: NVIDIA A10-8Q, Driver 570.211.01, CUDA 12.8

conda activate glass-cv
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Atteso: CUDA: True | GPU: NVIDIA A10-8Q
```

### 10.2 Storage Account

```bash
# Lista container
az storage container list \
  --account-name stglasscleanliness \
  --auth-mode key --output table
```

### 10.3 CVAT

```bash
# Verifica che il container Docker sia attivo sulla VM
az vm run-command invoke \
  -g kpmg-bologna -n BOLOGNA-AI-AM-MACHINE \
  --command-id RunShellScript \
  --scripts "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

Accesso da browser: `http://20.9.194.236:8080`  
Credenziali default CVAT: `admin` / (password impostata durante setup)

### 9.4 MLflow

```bash
# Verifica servizio systemd
az vm run-command invoke \
  -g kpmg-bologna -n BOLOGNA-AI-AM-MACHINE \
  --command-id RunShellScript \
  --scripts "systemctl status mlflow --no-pager"
```

Accesso da browser: `http://20.9.194.236:5000`

### 9.5 Conda environment

```bash
# Dalla SSH sulla VM
conda activate glass-cv
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Atteso: 2.5.1+cu121  True
```

---

## 11. Stato Attuale VM (Aprile 2026)

Questa sezione riflette lo stato reale della VM dopo il provisioning manuale eseguito il 13 Aprile 2026.

| Componente | Stato | Dettaglio |
|---|---|---|
| VM online | ✅ | Running, IP `20.9.194.236` |
| SSH accesso | ✅ | `ssh azureuseram@20.9.194.236` funzionante |
| GPU driver | ✅ | NVIDIA 570.211.01 via Azure Extension `NvidiaGpuDriverLinux v1.9` |
| CUDA | ✅ | 12.8 rilevato da nvidia-smi |
| Conda | ✅ | `conda 26.1.1` su `/mnt/miniconda` |
| Env `glass-cv` | ✅ | Python 3.10, PyTorch 2.5.1+cu121, ultralytics 8.4.37, timm 1.0.26, mlflow 3.11.1 |
| Disco root `/` | ✅ | ~16GB liberi (era pieno al 100%, liberato rimuovendo vecchi pesi SAM2/DINO) |
| Disco dati `/mnt` | ✅ | 360GB quasi liberi |
| CVAT | ⏳ | Sorgente in `/mnt/cvat`, da avviare con `docker compose up` |
| MLflow server | ⏳ | Da avviare manualmente (vedi sotto) |
| Dataset su VM | ⏳ | Da trasferire via azcopy o Roboflow |

### Avvio MLflow

```bash
ssh azureuseram@20.9.194.236
conda activate glass-cv
mkdir -p /mnt/mlflow-data/{mlruns,artifacts}
nohup mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri /mnt/mlflow-data/mlruns \
  --default-artifact-root /mnt/mlflow-data/artifacts \
  > /mnt/mlflow-data/mlflow.log 2>&1 &
echo "MLflow avviato → http://20.9.194.236:5000"
```

### Note sul driver NVIDIA

Il driver è stato installato tramite l'estensione Azure `NvidiaGpuDriverLinux v1.9` (non apt/dkms manuale).
Il kernel di boot è bloccato a `6.17.0-1010-azure` tramite GRUB (`GRUB_DEFAULT` impostato manualmente) per garantire compatibilità con il modulo DKMS `nvidia-srv/570.211.01`.

> **Importante:** non eseguire `apt upgrade` senza verificare che il nuovo kernel sia compatibile con il driver installato.

---

## 12. Problemi Noti e Soluzioni

Questa sezione documenta i problemi incontrati durante il provisioning reale e le soluzioni adottate.

### P1 — Git Bash (MSYS) converte i path Unix in path Windows

**Sintomo:**  
`/tmp/setup.sh` diventa `C:/Program Files/Git/tmp/setup.sh` quando passato al CLI.

**Causa:**  
Git Bash su Windows (MINGW64) include MSYS che intercetta stringhe che iniziano con `/` e le converte in path Windows.

**Soluzione adottata:**  
Usare `MSYS_NO_PATHCONV=1` **solo inline** (non globalmente), oppure usare path relativi. Non impostare `MSYS_NO_PATHCONV=1` come variabile d'ambiente globale: rompe la risoluzione del comando `az`.

```bash
# ❌ SBAGLIATO — rompe az
export MSYS_NO_PATHCONV=1
az ...

# ✅ CORRETTO — solo per il singolo comando
MSYS_NO_PATHCONV=1 az vm run-command ...

# ✅ ALTERNATIVA — path relativo (MSYS non tocca path senza slash iniziale)
TMPFILE="myscript.sh"     # ← relativo, sicuro
```

### P2 — `az vm run-command --scripts` spezza la stringa sugli spazi

**Sintomo:**  
`--scripts "wget -q -O /path/file 'URL'"` restituisce `unrecognized arguments: -q -O /path/file 'URL'`

**Causa:**  
`--scripts` usa `nargs='+'` in argparse: interpreta ogni token separato da spazio come script separato, non come singola stringa.

**Soluzione adottata:**  
Passare l'URL via base64 per eliminare i caratteri problematici (`&`, `%`, `?`, `=`) dall'URL SAS, poi decodificarlo sulla VM:

```bash
B64=$(echo -n "$BLOB_URL" | base64 | tr -d '\n')
az vm run-command invoke \
  --scripts "echo ${B64} | base64 -d | wget -q -i - -O /home/azureuseram/setup_vm.sh"
```

### P3 — Storage Account in Resource Group sbagliato

**Sintomo:**  
`az storage account show --name stglasscleanliness` → not found.

**Causa:**  
Lo storage account era nel RG originale della VM (`BOLOGNA-AI-AM-MACHINE-1_group`) invece che in `kpmg-bologna`.

**Soluzione:**  
Spostato manualmente dal portale Azure (`Impostazioni → Sposta → Sposta in un altro gruppo di risorse`). Tutti gli script usano `--resource-group kpmg-bologna`.

### P4 — Autenticazione storage `--auth-mode login` falliva

**Sintomo:**  
`az storage container create ... --auth-mode login` → `AuthorizationPermissionMismatch`

**Causa:**  
L'account Azure AD non aveva il ruolo RBAC `Storage Blob Data Contributor` sullo storage account.

**Soluzione:**  
Usare `--auth-mode key` ovunque: usa la chiave dell'account storage (equivale a accesso root), nessun RBAC necessario.

### P5 — Driver NVIDIA non carica al boot (kernel mismatch)

**Sintomo:**  
`nvidia-smi` → `couldn't communicate with the NVIDIA driver` dopo un reboot.

**Causa:**  
Il kernel Azure si aggiorna automaticamente (`apt upgrade`). Il driver DKMS è compilato per il kernel precedente e non funziona su quello nuovo.

**Soluzione adottata:**  
Bloccare GRUB sul kernel `6.17.0-1010-azure` dove il driver è compilato e funzionante:

```bash
sudo sed -i 's/GRUB_DEFAULT=0/GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.17.0-1010-azure"/' /etc/default/grub
sudo update-grub
sudo reboot
```

**Soluzione definitiva (futura):** usare l'estensione Azure `NvidiaGpuDriverLinux` — gestisce automaticamente i kernel aggiornamenti.

### P6 — `az vm run-command` bloccato (Conflict)

**Sintomo:**  
`(Conflict) Run command extension execution is in progress.`

**Causa:**  
Un run-command precedente è rimasto bloccato nell'extension agent della VM.

**Soluzione:**  
Stop + Start della VM dal portale Azure (non un semplice restart). Questo resetta l'extension agent senza perdere dati su `/mnt`.

---

## 13. Prossimi Step

La Fase 0 è **completata**. L'ordine delle prossime azioni è:

```bash
# 1. Trasferisci il dataset su Storage e poi sulla VM
#    (vedi Sezione 8 — Trasferimento con AzCopy)

# 2. Avvia MLflow sulla VM
conda activate glass-cv
nohup mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri /mnt/mlflow-data/mlruns \
  > /mnt/mlflow-data/mlflow.log 2>&1 &

# 3. Avvia il training YOLOv8 (Fase 2)
#    Vedi: docs/phase1_yolo_training.md
```

Vedere documentazione specifica per ogni fase:
- [phase1_yolo_training.md](phase1_yolo_training.md) — Training YOLOv8 Window Detector
- [pipeline_documentation.md](pipeline_documentation.md) — Documentazione completa della pipeline
