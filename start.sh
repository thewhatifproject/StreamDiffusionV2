#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CONFIG
# ============================================================
ENV_NAME="sdiff2"
PYTHON_VERSION="3.10"
MINICONDA_DIR="$HOME/miniconda3"

INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

# Repo StreamDiffusionV2
REPO_URL="https://github.com/thewhatifproject/StreamDiffusionV2.git"
REPO_BRANCH="dev"
REPO_DIR="$HOME/StreamDiffusionV2"

# ============================================================
# 1. DOWNLOAD MINICONDA
# ============================================================
echo "[1/12] Download Miniconda..."
cd /tmp
wget -q "${INSTALLER_URL}" -O "${INSTALLER}"

# ============================================================
# 2. INSTALLAZIONE MINICONDA (BATCH)
# ============================================================
echo "[2/12] Installazione Miniconda..."
if [ ! -d "${MINICONDA_DIR}" ]; then
  bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"
else
  echo "Miniconda già presente in ${MINICONDA_DIR}"
fi

# ============================================================
# 3. CARICAMENTO CONDA NELLA SESSIONE CORRENTE
# ============================================================
echo "[3/12] Caricamento Conda..."
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
"${MINICONDA_DIR}/bin/conda" config --set auto_activate_base false >/dev/null 2>&1 || true

# ============================================================
# 4. ACCETTAZIONE TERMS OF SERVICE
# ============================================================
echo "[4/12] Accettazione Terms of Service Anaconda..."
"${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ============================================================
# 5. CREAZIONE ENV (IDEMPOTENTE)
# ============================================================
echo "[5/12] Creazione environment '${ENV_NAME}'..."
if "${MINICONDA_DIR}/bin/conda" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment '${ENV_NAME}' già esistente"
else
  "${MINICONDA_DIR}/bin/conda" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

# ============================================================
# 6. ATTIVAZIONE ENV
# ============================================================
echo "[6/12] Attivazione environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# ============================================================
# 8. UPGRADE TOOLING PYTHON
# ============================================================
echo "[8/12] Upgrade pip / setuptools / wheel..."
python -m pip install -U pip setuptools wheel

# ============================================================
# 9. INSTALL PYTORCH CUDA 12.4
# ============================================================
echo "[9/12] Install PyTorch (cu124)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# 10. INSTALL FLASH-ATTN
# ============================================================
echo "[10/12] Install flash_attn..."
pip install --no-build-isolation --no-deps flash_attn==2.7.4.post1

# ============================================================
# 11. CLONE STREAMDIFFUSIONV2 (DEV) + REQUIREMENTS
# ============================================================
echo "[11/12] Clone StreamDiffusionV2 (branch dev)..."
if [ ! -d "${REPO_DIR}" ]; then
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
else
  echo "Repository già presente, aggiorno..."
  cd "${REPO_DIR}"
  git fetch origin
  git checkout "${REPO_BRANCH}"
  git pull origin "${REPO_BRANCH}"
fi

cd "${REPO_DIR}"

# ============================================================
# 12. EDITABLE INSTALL (al posto di setup.py develop)
# ============================================================
echo "[12/12] pip install -e . (editable)..."
pip install -e . --no-deps

# ============================================================
# VERIFICA FINALE
# ============================================================
echo "--------------------------------------------------"
echo "Setup completato con successo"
echo "Conda env attivo: ${CONDA_DEFAULT_ENV}"
python --version
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
EOF
echo "--------------------------------------------------"