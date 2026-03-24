#!/bin/bash
# ===========================================
#   DiveSeg UIIS10K - One-Click Training
#   Clone repo → run this script → done
# ===========================================
set -e

# ---- Configuration (edit as needed) ----
GPU_ID="${GPU_ID:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
VERIFY_ONLY="${VERIFY_ONLY:-true}"     # true = stop after training starts, then cleanup
VERIFY_WAIT_SEC="${VERIFY_WAIT_SEC:-90}" # seconds to let training run before stopping

# ---- Auto-detect repo root ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "      DiveSeg UIIS10K Training Setup      "
echo "=========================================="
echo "Repo dir:  $SCRIPT_DIR"
echo "GPU:       $GPU_ID"
echo "Verify:    $VERIFY_ONLY"
echo "=========================================="

# ---- 0. Setup CUDA environment ----
if [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME=/usr/local/cuda-12.4
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ---- 1. Install uv if needed ----
if ! command -v uv &> /dev/null; then
    echo "[1/5] 'uv' not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[1/5] 'uv' is already installed."
fi

# ---- 2. Virtual Environment & Dependencies ----
echo "[2/5] Setting up virtual environment..."
uv venv .venv
source .venv/bin/activate

echo "Installing build prerequisites..."
uv pip install --python .venv "setuptools<76" wheel packaging

echo "Installing PyTorch (CUDA 12.1)..."
uv pip install --python .venv torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing OpenCV..."
uv pip install --python .venv opencv-python-headless

echo "Installing remaining requirements..."
uv pip install --python .venv -r requirements.txt

echo "Installing Detectron2..."
uv pip install --python .venv git+https://github.com/facebookresearch/detectron2.git --no-build-isolation

echo "Pinning NumPy < 2 (PyTorch 2.1.0 compatibility)..."
uv pip install --python .venv "numpy<2"

echo "Compiling Pixel Decoder Ops for Mask2Former..."
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd "$SCRIPT_DIR"

# Re-activate venv (make.sh may have disrupted it)
source .venv/bin/activate

# ---- 3. Download Dataset ----
DATA_DIR="$SCRIPT_DIR/data/UIIS10K"
echo "[3/5] Downloading UIIS10K dataset to $DATA_DIR ..."
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/annotations" ] || [ ! -d "$DATA_DIR/img" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download('LiamLian0727/UIIS10K', repo_type='dataset', local_dir='$DATA_DIR')"

    # The HF repo contains UIIS10K.zip with nested UIIS10K/ folder
    if [ -f "$DATA_DIR/UIIS10K.zip" ]; then
        echo "Extracting UIIS10K.zip..."
        unzip -q -o "$DATA_DIR/UIIS10K.zip" -d "$DATA_DIR"
        # Move from nested UIIS10K/ to data root
        if [ -d "$DATA_DIR/UIIS10K" ]; then
            mv "$DATA_DIR/UIIS10K/"* "$DATA_DIR/"
            rmdir "$DATA_DIR/UIIS10K"
        fi
        rm -f "$DATA_DIR/UIIS10K.zip"
        rm -rf "$DATA_DIR/.cache"
        echo "Dataset extracted successfully."
    fi
else
    echo "Dataset already exists, skipping download."
fi

# ---- 4. Download DINOv2 ViT-L Checkpoint ----
CKPT_DIR="$SCRIPT_DIR/checkpoints"
CKPT_FILE="$CKPT_DIR/dinov2_vitl14_pretrain.pth"
echo "[4/5] Downloading DINOv2 ViT-L Checkpoint to $CKPT_DIR ..."
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_FILE" ]; then
    wget -O "$CKPT_FILE" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
else
    echo "Checkpoint already exists, skipping download."
fi

# ---- 5. Run Training ----
echo "[5/5] Starting Training..."

# Set dynamic data root for dataset registration
export DIVESEG_DATA_ROOT="$DATA_DIR"

CONFIG_FILE="configs/UIIS/instance-segmentation/dinov2/dinov2_vit_large_UIIS10K.yaml"

if [ "$VERIFY_ONLY" = "true" ]; then
    echo "==> VERIFY MODE: Training will run for ${VERIFY_WAIT_SEC}s then stop + cleanup"

    # Run training in background
    python train_net.py --num-gpus "$NUM_GPUS" \
        --config-file "$CONFIG_FILE" \
        --resume &
    TRAIN_PID=$!

    echo "Training PID: $TRAIN_PID"
    echo "Waiting ${VERIFY_WAIT_SEC}s for training to start..."
    sleep "$VERIFY_WAIT_SEC"

    # Check if training is still running (= it started successfully)
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "=========================================="
        echo "  ✅ Training started successfully!"
        echo "=========================================="
        echo "Stopping training and cleaning up..."
        kill "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    else
        echo "=========================================="
        echo "  ❌ Training process exited prematurely!"
        echo "=========================================="
        wait "$TRAIN_PID" 2>/dev/null || true
        EXIT_CODE=$?
        echo "Exit code: $EXIT_CODE"
    fi

    # ---- Cleanup ----
    echo "Cleaning up: removing .venv, data, checkpoints, out_put..."
    deactivate 2>/dev/null || true
    rm -rf "$SCRIPT_DIR/.venv"
    rm -rf "$SCRIPT_DIR/data"
    rm -rf "$SCRIPT_DIR/checkpoints"
    rm -rf "$SCRIPT_DIR/out_put"
    echo "=========================================="
    echo "  🧹 Cleanup complete! Repo is clean."
    echo "=========================================="
else
    echo "==> FULL TRAINING MODE"
    python train_net.py --num-gpus "$NUM_GPUS" \
        --config-file "$CONFIG_FILE" \
        --resume
fi

echo "Done!"
