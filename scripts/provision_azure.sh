#!/bin/bash
# =============================================================================
# provision_azure.sh — Provisioning risorse Azure (versione Bash)
#
# ESECUZIONE:
#   bash scripts/provision_azure.sh
# =============================================================================

set -euo pipefail

AZ="az"

RG_STORAGE="kpmg-bologna"                      # Resource Group dello Storage Account
RG_VM="kpmg-bologna"                           # Resource Group della VM
STORAGE_ACCOUNT="stglasscleanliness"          # già creato
VM_NAME="BOLOGNA-AI-AM-MACHINE"               # già esistente in kpmg-bologna / westus2
ADMIN_USER="azureuseram"

echo "================================================================"
echo " Glass Cleanliness — Provisioning Azure Resources"
echo "================================================================"

# ── 0. Verifica login ────────────────────────────────────────────────────────
echo -e "\n[0] Verifica login Azure..."
if ! $AZ account show &>/dev/null; then
    echo "  Non sei loggato. Eseguo $AZ login..."
    $AZ login
fi
ACCOUNT_NAME=$($AZ account show --query name --output tsv)
echo "  Account: $ACCOUNT_NAME"

# ── 1. Verifica Resource Group esistenti ────────────────────────────────────
echo -e "\n[1/3] Verifica Resource Groups..."
LOCATION_STORAGE=$($AZ group show --name "$RG_STORAGE" --query location --output tsv 2>/dev/null)
if [ -z "$LOCATION_STORAGE" ]; then
    echo "  ERRORE: Resource Group storage '$RG_STORAGE' non trovato."
    exit 1
fi
echo "  OK — $RG_STORAGE → $LOCATION_STORAGE"

LOCATION_VM=$($AZ group show --name "$RG_VM" --query location --output tsv 2>/dev/null)
if [ -z "$LOCATION_VM" ]; then
    echo "  ERRORE: Resource Group VM '$RG_VM' non trovato."
    exit 1
fi
echo "  OK — $RG_VM → $LOCATION_VM"

# ── 2. Storage Account ───────────────────────────────────────────────────────
echo -e "\n[2/3] Storage Account: $STORAGE_ACCOUNT..."

CURRENT_RG=$($AZ storage account show --name "$STORAGE_ACCOUNT" --query resourceGroup --output tsv 2>/dev/null || echo "")
if [ -z "$CURRENT_RG" ]; then
    echo "  ERRORE: Storage account '$STORAGE_ACCOUNT' non trovato in nessun RG."
    exit 1
fi
echo "  Trovato in: $CURRENT_RG"

# Crea i container se non esistono già
echo "  Verifica/creazione container Blob..."
for container in "raw-images" "annotations" "models" "results"; do
    $AZ storage container create \
        --name "$container" \
        --account-name "$STORAGE_ACCOUNT" \
        --auth-mode key \
        --output none 2>/dev/null && echo "    + $container" || echo "    ~ $container (già esistente)"
done

CONN_STR=$($AZ storage account show-connection-string \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RG_STORAGE" \
    --query connectionString \
    --output tsv)

echo "AZURE_STORAGE_CONNECTION_STRING=$CONN_STR" > .env.local
echo "  Connection string salvata in .env.local"

# ── 3. VM esistente (kpmg-bologna / westus2): verifica + apertura porte ──────
echo -e "\n[3/3] Verifica VM esistente: $VM_NAME (RG: $RG_VM)..."
VM_SIZE=$($AZ vm show \
    --resource-group "$RG_VM" \
    --name "$VM_NAME" \
    --query hardwareProfile.vmSize \
    --output tsv 2>/dev/null)

if [ -z "$VM_SIZE" ]; then
    echo "  ERRORE: VM '$VM_NAME' non trovata in '$RG_VM'."
    exit 1
fi
echo "  OK — Size: $VM_SIZE"

echo "  Apertura porte NSG (skip se già aperte)..."
$AZ vm open-port --resource-group "$RG_VM" --name "$VM_NAME" \
    --port 8080 --priority 1001 --output none 2>/dev/null && echo "    + 8080 (CVAT)" || echo "    ~ 8080 già aperta"
$AZ vm open-port --resource-group "$RG_VM" --name "$VM_NAME" \
    --port 5000 --priority 1002 --output none 2>/dev/null && echo "    + 5000 (MLflow)" || echo "    ~ 5000 già aperta"

VM_IP=$($AZ vm show \
    --resource-group "$RG_VM" \
    --name "$VM_NAME" \
    --show-details \
    --query publicIps \
    --output tsv)

echo "VM_PUBLIC_IP=$VM_IP" >> .env.local
echo "  IP pubblico VM: $VM_IP"

# ── Trasferimento setup_vm.sh: upload su Blob + run-command INLINE ────────────
# --scripts accetta stringhe inline (lo stesso modo con cui funzionano ls/whoami).
# Il SAS URL viene costruito in bash e passato direttamente come stringa.
# Nessun @file → nessun problema di path mangling su Git Bash Windows.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "\nCaricamento setup_vm.sh su Blob Storage..."

$AZ storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "models" \
    --name "scripts/setup_vm.sh" \
    --file "${SCRIPT_DIR}/setup_vm.sh" \
    --auth-mode key \
    --overwrite \
    --output none

# Genera SAS URL valido 2 ore (serve solo per il download, poi scade)
EXPIRY=$(date -u -d "+2 hours" '+%Y-%m-%dT%H:%MZ' 2>/dev/null \
    || date -u -v+2H '+%Y-%m-%dT%H:%MZ')

SAS_TOKEN=$($AZ storage blob generate-sas \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "models" \
    --name "scripts/setup_vm.sh" \
    --permissions r \
    --expiry "$EXPIRY" \
    --auth-mode key \
    --output tsv)

BLOB_URL="https://${STORAGE_ACCOUNT}.blob.core.windows.net/models/scripts/setup_vm.sh?${SAS_TOKEN}"

echo "  Invio wget alla VM tramite run-command inline..."
# Nota: --scripts ACCETTA stringhe inline (verificato con ls/whoami/tail).
# Il doppio apice nella stringa bash non crea problemi perché
# az gestisce internamente la serializzazione verso l'API ARM.
$AZ vm run-command invoke \
    --resource-group "$RG_VM" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts "wget -q -O /home/${ADMIN_USER}/setup_vm.sh '${BLOB_URL}' && chmod +x /home/${ADMIN_USER}/setup_vm.sh && echo 'DOWNLOAD_OK'" \
    --query "value[0].message" \
    --output tsv

# Verifica che il file sia arrivato
echo "  Verifica presenza file sulla VM..."
$AZ vm run-command invoke \
    --resource-group "$RG_VM" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts "ls -lh /home/${ADMIN_USER}/setup_vm.sh && echo 'FILE_PRESENTE'" \
    --query "value[0].message" \
    --output tsv

# ── Riepilogo ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Provisioning completato!"
echo "================================================================"
echo "  RG Storage     : $RG_STORAGE"
echo "  RG VM          : $RG_VM"
echo "  Storage Account: $STORAGE_ACCOUNT"
echo "  VM IP          : $VM_IP"
echo ""
echo "  Prossimi step:"
echo "    1. Avvia il setup sulla VM (via run-command, nessun SSH necessario):"
echo "         $AZ vm run-command invoke -g $RG_VM -n $VM_NAME --command-id RunShellScript --scripts 'bash /home/${ADMIN_USER}/setup_vm.sh > /home/${ADMIN_USER}/setup.log 2>&1 &'"
echo ""
echo "    2. Monitora il progresso:"
echo "         $AZ vm run-command invoke -g $RG_VM -n $VM_NAME --command-id RunShellScript --scripts 'tail -50 /home/${ADMIN_USER}/setup.log'"
echo ""
echo "    3. Copia .env.local -> .env e aggiungi MAPILLARY_ACCESS_TOKEN"
echo "================================================================"