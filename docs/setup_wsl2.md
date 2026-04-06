# WSL2 Setup for RelayKV

## Goal

Set up a minimal WSL2-based development environment for RelayKV experiments.

## Host Environment

- Windows 11
- WSL2
- Ubuntu
- NVIDIA GPU on Windows host

## 1. Install WSL2

Run the following in PowerShell as Administrator:

```powershell
wsl --install
wsl --update
```

Reboot if needed.

## 2. Prepare Ubuntu

Open Ubuntu and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget tmux htop
sudo apt install -y python3 python3-pip python3-venv
```

## 3. Create a workspace

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/rinsakamo/relay-kv.git
cd relay-kv
```

## 4. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 5. Install PyTorch

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 6. Install experiment dependencies

```bash
pip install transformers accelerate sentencepiece safetensors numpy pandas matplotlib psutil
```

## 7. Verify GPU access

```bash
nvidia-smi
```

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## Notes

- Use the WSL filesystem for active development.
- Keep the experiment environment fixed for reproducibility.
- Start with PyTorch + Transformers before trying other runtimes.
