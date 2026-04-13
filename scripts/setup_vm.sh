#!/bin/bash
# =============================================================================
# setup_vm.sh — Setup completo della VM all-in-one su Azure
#
# ESECUZIONE (via az vm run-command):
#   az vm run-command invoke -g kpmg-bologna -n BOLOGNA-AI-AM-MACHINE \
#     --command-id RunShellScript --scripts "bash /tmp/setup_vm.sh > /tmp/setup.log 2>&1"
#
# Al termine:
#   CVAT   → http://<VM_IP>:8080
#   MLflow → http://<VM_IP>:5000
# =============================================================================

set -euo pipefail
HOME="${HOME:-/home/azureuseram}"
export HOME
export DEBIAN_FRONTEND=noninteractive

CONDA_ENV="glass-cv"
PYTHON_VERSION="3.11"
MINICONDA_DIR="/mnt/miniconda"
WEIGHTS_DIR="/mnt/weights"
CVAT_DIR="/mnt/cvat"
MLFLOW_DIR="/mnt/mlflow-data"

echo "================================================================"
echo " Glass Cleanliness Detection — VM Setup"
echo "================================================================"

# =============================================================================
# STEP 0 — Prepara /mnt con tutte le directory necessarie
# (tutto il dato pesante va su /mnt, il root disk è solo 29GB)
# =============================================================================
echo "[0/6] Preparazione /mnt..."
sudo mkdir -p \
    /mnt/docker-data \
    /mnt/tmp \
    /mnt/pip-cache \
    "$WEIGHTS_DIR" \
    "$MLFLOW_DIR/artifacts" \
    "$CVAT_DIR"
sudo chmod 1777 /mnt/tmp
sudo chown -R "$USER:$USER" "$WEIGHTS_DIR" "$MLFLOW_DIR" "$CVAT_DIR" 2>/dev/null || true
export TMPDIR="/mnt/tmp"
export PIP_CACHE_DIR="/mnt/pip-cache"
echo "  /mnt pronto."

# -----------------------------------------------------------------------------
# 1. Aggiorna sistema e installa dipendenze di sistema
# -----------------------------------------------------------------------------
echo "[1/6] Aggiornamento sistema..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl wget unzip rsync \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1

# -----------------------------------------------------------------------------
# 2. Docker + Docker Compose (per CVAT)
#    TUTTO su /mnt: docker-data (daemon.json) + containerd (symlink)
# -----------------------------------------------------------------------------
echo "[2/6] Installazione Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installato."
else
    echo "  Docker già installato, skip."
fi

if ! docker compose version &>/dev/null 2>&1; then
    sudo apt-get install -y -qq docker-compose-plugin
fi

# ── Docker data-root → /mnt/docker-data ─────────────────────────────────────
echo "  Configurazione Docker data-root su /mnt/docker-data..."
echo '{"data-root":"/mnt/docker-data"}' | sudo tee /etc/docker/daemon.json > /dev/null

# ── Containerd → /mnt/containerd (symlink — il config.toml viene spesso ignorato) ──
echo "  Migrazione containerd su /mnt/containerd (symlink)..."
sudo systemctl stop docker 2>/dev/null || true
sudo systemctl stop containerd 2>/dev/null || true
sleep 2

# Se /var/lib/containerd è una directory reale (non symlink), migra i dati
if [ ! -L /var/lib/containerd ] && [ -d /var/lib/containerd ]; then
    echo "  Migrazione dati containerd esistenti..."
    sudo mkdir -p /mnt/containerd
    sudo rsync -a /var/lib/containerd/ /mnt/containerd/ 2>/dev/null || true
    sudo rm -rf /var/lib/containerd
fi

# Crea symlink se non esiste già
if [ ! -L /var/lib/containerd ]; then
    sudo mkdir -p /mnt/containerd
    sudo ln -s /mnt/containerd /var/lib/containerd
fi

sudo systemctl start containerd
sleep 3
sudo systemctl start docker
sleep 3

echo "  Docker root: /mnt/docker-data"
echo "  Containerd:  /var/lib/containerd -> /mnt/containerd (symlink)"
ls -la /var/lib/containerd | head -1
df -h /dev/root /mnt

# -----------------------------------------------------------------------------
# 3. Miniconda + ambiente Python
# -----------------------------------------------------------------------------
echo "[3/6] Setup Conda e ambiente Python $PYTHON_VERSION..."
if [ ! -d "$MINICONDA_DIR" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /mnt/tmp/miniconda.sh
    bash /mnt/tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f /mnt/tmp/miniconda.sh
else
    echo "  Miniconda già installata in $MINICONDA_DIR, skip."
fi

# Symlink da $HOME/miniconda per compatibilità
if [ ! -L "$HOME/miniconda" ] && [ ! -d "$HOME/miniconda" ]; then
    ln -s "$MINICONDA_DIR" "$HOME/miniconda"
fi

eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"

# Accetta TOS (richiesto dalla v24+)
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority strict 2>/dev/null || true

# Crea env solo se non esiste
if ! conda env list | grep -q "^$CONDA_ENV "; then
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y --quiet
fi
conda activate "$CONDA_ENV"

# Pulizia cache prima di installare
"$MINICONDA_DIR/bin/conda" clean --all -f -y -q 2>/dev/null || true
sudo apt-get clean -qq

# Forza pip a usare /mnt (evita "No space left" sul disco root)
export TMPDIR="/mnt/tmp"
export PIP_CACHE_DIR="/mnt/pip-cache"

# PyTorch con CUDA 12.1 (A10 su Azure)
echo "  Installazione PyTorch con CUDA 12.1..."
pip install --quiet --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Dipendenze principali
echo "  Installazione dipendenze Python..."
pip install --quiet --no-cache-dir \
    pytorch-lightning>=2.2.0 \
    timm>=0.9.16 \
    transformers>=4.38.0 \
    grad-cam>=1.5.0 \
    opencv-python-headless>=4.9.0 \
    Pillow>=10.2.0 \
    albumentations>=1.4.0 \
    supervision>=0.19.0 \
    imagehash>=4.3.1 \
    scikit-learn>=1.4.0 \
    scipy>=1.12.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    mlflow>=2.11.0 \
    azure-storage-blob>=12.19.0 \
    azure-identity>=1.15.0 \
    fastapi>=0.110.0 \
    uvicorn>=0.27.0 \
    python-multipart>=0.0.9 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0.1 \
    python-dotenv>=1.0.1 \
    roboflow>=1.1.0 \
    matplotlib>=3.8.0

pip install --quiet --no-cache-dir git+https://github.com/openai/CLIP.git
pip install --quiet --no-cache-dir groundingdino-py
pip install --quiet --no-cache-dir git+https://github.com/facebookresearch/sam2.git

echo "  Ambiente Python pronto."

# -----------------------------------------------------------------------------
# 4. Download pesi pre-addestrati su /mnt/weights
# -----------------------------------------------------------------------------
echo "[4/6] Download model weights → $WEIGHTS_DIR..."
mkdir -p "$WEIGHTS_DIR"

if [ ! -f "$WEIGHTS_DIR/sam2.1_hiera_large.pt" ]; then
    echo "  Downloading SAM2 Large (~900MB)..."
    wget -q --show-progress -P "$WEIGHTS_DIR" \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "  SAM2 weights già presenti, skip."
fi

if [ ! -f "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "  Downloading Grounding DINO SwinT (~700MB)..."
    wget -q --show-progress -P "$WEIGHTS_DIR" \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "  Grounding DINO weights già presenti, skip."
fi

if [ ! -f "$WEIGHTS_DIR/GroundingDINO_SwinT_OGC.py" ]; then
    wget -q -P "$WEIGHTS_DIR" \
        https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
fi

# -----------------------------------------------------------------------------
# 5. CVAT via Docker Compose (tutto su /mnt/cvat)
# -----------------------------------------------------------------------------
echo "[5/6] Setup CVAT..."
if [ ! -d "$CVAT_DIR/.git" ]; then
    git clone --quiet https://github.com/cvat-ai/cvat.git "$CVAT_DIR"
fi

cd "$CVAT_DIR"

# Pull immagini (può richiedere 30-60 minuti, da lanciare separatamente se necessario)
echo "  Pull immagini CVAT (potrebbe richiedere tempo)..."
sudo docker compose pull

# Avvia CVAT
echo "  Avvio CVAT..."
sudo docker compose up -d

# Attende che il server sia pronto (max 60s)
echo "  Attendo che CVAT sia pronto..."
for i in $(seq 1 12); do
    sleep 5
    if sudo docker exec cvat_server python manage.py check 2>/dev/null; then
        break
    fi
done

# Crea superuser admin (ignora se già esiste)
echo "  Creazione admin CVAT..."
sudo docker exec cvat_server bash -c \
    'DJANGO_SUPERUSER_PASSWORD=admin123 python manage.py createsuperuser \
     --username admin --email admin@glass.ai --noinput 2>/dev/null || true'

VM_IP=$(curl -s ifconfig.me 2>/dev/null || echo "VM_IP")
echo "  CVAT disponibile su: http://${VM_IP}:8080"
echo "  Credenziali: admin / admin123 (CAMBIA IN PRODUZIONE)"
cd "$HOME"

# -----------------------------------------------------------------------------
# 6. MLflow tracking server (systemd service) su /mnt/mlflow-data
# -----------------------------------------------------------------------------
echo "[6/6] Setup MLflow server..."

mkdir -p "$MLFLOW_DIR/artifacts"

sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=$USER
WorkingDirectory=$MLFLOW_DIR
Environment=TMPDIR=/mnt/tmp
ExecStart=$MINICONDA_DIR/envs/$CONDA_ENV/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///$MLFLOW_DIR/mlflow.db \
    --default-artifact-root $MLFLOW_DIR/artifacts
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow

VM_IP=$(curl -s ifconfig.me 2>/dev/null || echo "VM_IP")
echo "  MLflow disponibile su: http://${VM_IP}:5000"

# -----------------------------------------------------------------------------
# Riepilogo
# -----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Setup completato!"
echo "================================================================"
echo ""
echo "  CVAT (annotazioni)  → http://${VM_IP}:8080"
echo "                         user: admin / password: admin123"
echo ""
echo "  MLflow (tracking)   → http://${VM_IP}:5000"
echo ""
echo "  Ambiente Python:      conda activate $CONDA_ENV"
echo "  Pesi modelli:         $WEIGHTS_DIR"
echo "  CVAT files:           $CVAT_DIR"
echo "  MLflow data:          $MLFLOW_DIR"
echo ""
echo "  Disco /mnt:"
df -h /mnt
echo "  Disco / (root):"
df -h /dev/root
echo "================================================================"
