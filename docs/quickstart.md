# Quickstart

## Requirements

- **Python 3.12+**
- **[Node.js 18+](https://nodejs.org/)** (required to build the frontend)
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **LSL binaries** (liblsl) — platform-specific, see below

## Install LSL Binaries

::: code-group
```bash [Windows]
# Download latest liblsl from https://github.com/sccn/liblsl/releases
# Extract .dll files to C:\Windows\System32
```

```bash [macOS]
brew install labstreaminglayer/tap/lsl

# If pylsl can't find the library:
export PYLSL_LIB=$(brew --prefix lsl)/lib/liblsl.dylib
```

```bash [Linux (Ubuntu 24.04)]
# Download liblsl-1.16.2-jammy_amd64.deb from:
# https://github.com/sccn/liblsl/releases/tag/v1.16.2
cd ~/Downloads
sudo dpkg -i liblsl-1.16.2-jammy_amd64.deb
```
:::

::: warning
LSL binaries must be installed **before** `uv sync`.
:::

## 1. Clone and Install

::: code-group
```bash [Linux / macOS]
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/dendrite-bci/dendrite.git
cd dendrite
uv sync
```

```powershell [Windows]
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/dendrite-bci/dendrite.git
cd dendrite
uv sync
```
:::

## 2. Build the Frontend

```bash
cd frontend
npm install
npm run build
```

## 3. Start the Server

```bash
uv run dendrite --host 0.0.0.0 --port 8321
```

Open `http://localhost:8321`. LAN access at `http://<your-ip>:8321`. API docs at `/docs`.

## Optional

### Development Mode

For frontend hot-reload during development, run Vite instead of building:

```bash
cd frontend && npm run dev
```

Dev server at `http://localhost:5173`, proxies API requests to the backend.

### GPU Acceleration

::: code-group
```bash [NVIDIA (Linux/Windows)]
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

```bash [macOS (Apple Silicon)]
# MPS acceleration works automatically
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```
:::

## Next Steps

- [Guides](/guides/) — Recording, training, and deployment tutorials
- [Architecture](/architecture/) — How the system works
