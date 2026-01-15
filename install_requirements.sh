#!/bin/bash

# Installation script for StreamDiffusionV2 dependencies
# This script handles the flash_attn build dependency issue

set -e  # Exit on error

echo "========================================="
echo "Installing StreamDiffusionV2 dependencies"
echo "========================================="

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
