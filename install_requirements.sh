#!/bin/bash

# Installation script for StreamDiffusionV2 dependencies
# This script handles the flash_attn build dependency issue

set -e  # Exit on error

echo "========================================="
echo "Installing StreamDiffusionV2 dependencies"
echo "========================================="

# Step 0: Install Miniconda and create a dedicated env
echo ""
echo "Step 0: Installing Miniconda and creating env..."

ENV_NAME="streamdiffusionv2"
MINICONDA_DIR="$HOME/miniconda3"

if ! command -v conda >/dev/null 2>&1; then
    if [ ! -d "$MINICONDA_DIR" ]; then
        MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
        if [ "$(uname)" = "Darwin" ]; then
            MINICONDA_INSTALLER="/tmp/Miniconda3-latest-MacOSX-x86_64.sh"
        fi

        echo "Downloading Miniconda installer..."
        curl -fsSL "https://repo.anaconda.com/miniconda/$(basename "$MINICONDA_INSTALLER")" -o "$MINICONDA_INSTALLER"
        echo "Installing Miniconda to $MINICONDA_DIR..."
        bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
    fi

    # shellcheck disable=SC1091
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

# Step 1: Install PyTorch and related packages first (if not already installed)
echo ""
echo "Step 1: Ensuring PyTorch is installed..."
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    triton==3.2.0

# Step 2: Install flash_attn with no-build-isolation
# This allows flash_attn to access the already-installed torch during build
echo ""
echo "Step 2: Installing flash_attn (this may take a few minutes)..."
pip install flash_attn==2.7.4.post1 --no-build-isolation

# Step 3: Install remaining requirements
echo ""
echo "Step 3: Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
