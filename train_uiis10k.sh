#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================="
echo "      DiveSeg UIIS10K Training Setup      "
echo "=========================================="

# 0. Setup CUDA environment
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "[1/5] 'uv' not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add cargo bin to path for current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[1/5] 'uv' is already installed."
fi

export UV_VENV_CLEAR=1

# 2. Setup Virtual Environment & Install Requirements
echo "[2/5] Setting up virtual environment..."
uv venv .venv
source .venv/bin/activate

echo "Installing build prerequisites..."
uv pip install setuptools==69.5.1 wheel packaging

echo "Installing PyTorch (CUDA 12.1)..."
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining requirements..."
uv pip install -r requirements.txt

echo "Installing Detectron2..."
# Ensuring CUDA_HOME is set before pip install detectron2
uv pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation

# Compile custom ops for Mask2Former
echo "Compiling Pixel Decoder Ops for Mask2Former..."
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..

# 3. Download Dataset
echo "[3/5] Downloading UIIS10K dataset..."
mkdir -p data/UIIS
huggingface-cli download LiamLian0727/UIIS10K --repo-type dataset --local-dir data/UIIS

# 4. Download DINOv2 Checkpoint
echo "[4/5] Downloading DINOv2 ViT-L Checkpoint..."
mkdir -p checkpoints
if [ ! -f "checkpoints/dinov2_vitl14_pretrain.pth" ]; then
    wget -O checkpoints/dinov2_vitl14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
else
    echo "Checkpoint already exists."
fi

# 5. Run Training
echo "[5/5] Starting Training..."

# Optimal thread pinning
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
taskset -c 0-9 \
python train_net.py --num-gpus 1 \
  --config-file configs/UIIS/instance-segmentation/dinov2/dinov2_vit_large_UIIS10K.yaml \
  --resume
