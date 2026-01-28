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
echo "[1/15] Download Miniconda..."
cd /tmp
wget -q "${INSTALLER_URL}" -O "${INSTALLER}"

# ============================================================
# 2. INSTALLAZIONE MINICONDA
# ============================================================
echo "[2/15] Installazione Miniconda..."
if [ ! -d "${MINICONDA_DIR}" ]; then
  bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"
fi

# ============================================================
# 3. CARICAMENTO CONDA (SESSIONE CORRENTE)
# ============================================================
echo "[3/15] Caricamento Conda..."
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
"${MINICONDA_DIR}/bin/conda" config --set auto_activate_base false >/dev/null 2>&1 || true

# ============================================================
# 4. AUTO-ACTIVATE SEMPRE sdiff2 (SSH SAFE)
# ============================================================
echo "[4/15] Configurazione auto-activate shell..."

# login shell → carica bashrc
if ! grep -q 'source ~/.bashrc' "$HOME/.bash_profile" 2>/dev/null; then
  cat >> "$HOME/.bash_profile" <<'EOF'

if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
EOF
fi

# conda + activate
if ! grep -q 'conda activate sdiff2' "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF

# --- Conda ---
if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
  . "$MINICONDA_DIR/etc/profile.d/conda.sh"
fi
conda activate sdiff2
EOF
fi

# ============================================================
# 5. ACCETTAZIONE TERMS OF SERVICE
# ============================================================
echo "[5/15] Accettazione ToS Anaconda..."
"${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ============================================================
# 6. CREAZIONE ENV
# ============================================================
echo "[6/15] Creazione environment '${ENV_NAME}'..."
if ! "${MINICONDA_DIR}/bin/conda" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${MINICONDA_DIR}/bin/conda" create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# ============================================================
# 7. NODE.JS 18 via NVM
# ============================================================
echo "[7/15] Installazione Node.js 18 (nvm)..."

export NVM_DIR="$HOME/.nvm"

if [ ! -d "$NVM_DIR" ]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi

# carica nvm nella sessione corrente
# shellcheck disable=SC1090
source "$NVM_DIR/nvm.sh"

# install / usa / default
nvm install 18
nvm use 18
nvm alias default 18

# garantisce nvm a ogni SSH
if ! grep -q 'NVM_DIR' "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# --- NVM ---
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
EOF
fi

node -v
npm -v

# ============================================================
# 8. CLOUDFLARED (Ubuntu)
# ============================================================
echo "[8/15] Installazione cloudflared (Ubuntu)..."

if command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared già installato"
elif command -v apt-get >/dev/null 2>&1; then
  UBUNTU_CODENAME=""
  if command -v lsb_release >/dev/null 2>&1; then
    UBUNTU_CODENAME="$(lsb_release -cs)"
  elif [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    UBUNTU_CODENAME="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
  fi

  if [ -z "${UBUNTU_CODENAME}" ]; then
    echo "Impossibile rilevare codename Ubuntu: salta installazione cloudflared"
  else
    sudo mkdir -p /usr/share/keyrings
    if [ ! -f /usr/share/keyrings/cloudflare-main.gpg ]; then
      curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
    fi
    if [ ! -f /etc/apt/sources.list.d/cloudflared.list ]; then
      echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared ${UBUNTU_CODENAME} main" \
        | sudo tee /etc/apt/sources.list.d/cloudflared.list >/dev/null
    fi
    sudo apt-get update
    sudo apt-get install -y cloudflared
  fi
else
  echo "apt-get non disponibile: salta installazione cloudflared"
fi

# ============================================================
# 9. TOOLING PYTHON
# ============================================================
echo "[9/15] Upgrade pip / setuptools / wheel..."
python -m pip install -U pip setuptools wheel

# ============================================================
# 10. PYTORCH CUDA 12.4
# ============================================================
echo "[10/15] Install PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# 11. PSUTIL + FLASH-ATTN
# ============================================================
echo "[11/15] Install psutil + flash_attn..."
pip install psutil
pip install --no-build-isolation --no-deps flash_attn==2.7.4.post1

# ============================================================
# 12. CLONE STREAMDIFFUSIONV2
# ============================================================
echo "[12/15] Clone StreamDiffusionV2..."
if [ ! -d "${REPO_DIR}" ]; then
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

# ============================================================
# 13. REQUIREMENTS
# ============================================================
if grep -qE '^\s*nvidia-pyindex\b' requirements.txt; then
  sed -i '/^\s*nvidia-pyindex\b/d' requirements.txt
fi
pip install -r requirements.txt --no-deps
pip install -e . --no-deps

# ============================================================
# 14. DOWNLOAD MODELLI
# ============================================================
echo "[14/15] Download modelli..."
pip install -U huggingface_hub

huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include "wan_causal_dmd_v2v/*"

# ============================================================
# 15. VERIFICA FINALE
# ============================================================
echo "--------------------------------------------------"
echo "Setup completato"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
node -v
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
EOF
echo "--------------------------------------------------"
