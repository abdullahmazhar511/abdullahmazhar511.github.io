set -euo pipefail

# --- Layout ---------------------------------------------------------------
mkdir -p /workspace/tools/{bin,envs,logs} /root/.config
cd /workspace/tools

# --- System deps (for OpenCV, pycocotools, onnxruntime, git) --------------
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl wget ca-certificates build-essential pkg-config \
  python3-dev ffmpeg libgl1 libglib2.0-0 git-lfs
git lfs install

# --- Micromamba (tiny conda) ---------------------------------------------
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
 | tar -xvj -C /workspace/tools bin/micromamba

# Shell init (persistent)
cat >/workspace/tools/mamba_init.sh <<'EOF'
export MAMBA_ROOT_PREFIX=/workspace/tools/micromamba
eval "$(/workspace/tools/bin/micromamba shell hook -s bash)"
EOF

# Add to bashrc if not present
grep -q 'mamba_init.sh' ~/.bashrc || echo 'source /workspace/tools/mamba_init.sh' >> ~/.bashrc
source /workspace/tools/mamba_init.sh

# --- Env YAMLs ------------------------------------------------------------
cat > /workspace/tools/envs/ml.yml <<'YAML'
name: ml
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - matplotlib
  - tqdm
  - jupyterlab
  - ipykernel
  - pillow
  - scikit-image
  - scikit-learn
  - pip:
      # PyTorch CUDA 12.1 wheels include CUDA runtime
      - --extra-index-url https://download.pytorch.org/whl/cu121
      - torch
      - torchvision
      - torchaudio
      - xformers
      - transformers
      - accelerate
      - peft
      - datasets
      - timm
      - sentencepiece
      - bitsandbytes
      - huggingface_hub
      - opencv-python-headless
YAML

cat > /workspace/tools/envs/vision.yml <<'YAML'
name: vision
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip
  - jupyterlab
  - ipykernel
  - numpy
  - pillow
  - scikit-learn
  - pip:
      - opencv-python-headless
      - onnx
      - onnxruntime-gpu
      - insightface==0.7.3
YAML

cat > /workspace/tools/envs/sam.yml <<'YAML'
name: sam
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip
  - jupyterlab
  - ipykernel
  - matplotlib
  - numpy
  - pillow
  - pip:
      - opencv-python-headless
      - pycocotools
      - git+https://github.com/facebookresearch/segment-anything.git
YAML

# --- Helper scripts -------------------------------------------------------
cat > /workspace/tools/bin/rebuild_envs.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/mamba_init.sh
for f in /workspace/tools/envs/*.yml; do
  echo "==> Creating env from $f"
  /workspace/tools/bin/micromamba create -y -f "$f"
  ENV_NAME=$(python - <<PY
import sys, yaml, pathlib
print(yaml.safe_load(pathlib.Path("$f").read_text())["name"])
PY
)
  echo "==> Registering Jupyter kernel: $ENV_NAME"
  /workspace/tools/bin/micromamba run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
done
echo "All envs built."
EOF
chmod +x /workspace/tools/bin/rebuild_envs.sh

cat > /workspace/tools/bin/use <<'EOF'
#!/usr/bin/env bash
source /workspace/tools/mamba_init.sh
if [ $# -lt 1 ]; then
  echo "Usage: use <env-name>"
  exit 1
fi
/mnt/bin/true 2>/dev/null || true  # no-op placeholder
micromamba activate "$1"
EOF
chmod +x /workspace/tools/bin/use

cat > /workspace/tools/bin/test_gpu.py <<'EOF'
import torch, sys
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    a = torch.randn(1024,1024, device='cuda')
    b = torch.randn(1024,1024, device='cuda')
    c = (a @ b).sum().item()
    print("Matmul OK, checksum:", c)
EOF

# --- Build envs -----------------------------------------------------------
/workspace/tools/bin/rebuild_envs.sh

# --- Quick GPU sanity for ml env -----------------------------------------
/workspace/tools/bin/micromamba run -n ml python /workspace/tools/bin/test_gpu.py | tee /workspace/tools/logs/gpu_check.txt

echo
echo "Setup complete. To start using an env now:"
echo "  source /workspace/tools/mamba_init.sh"
echo "  micromamba activate ml    # or: /workspace/tools/bin/use ml"
