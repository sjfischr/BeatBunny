# BeatBunny 🐇🎛️
A local-first, open-source music generation studio — aiming to be the **AUTOMATIC1111 for music**.

BeatBunny wraps **HeartMuLa-oss-3B** into a simple web UI so you can generate songs from **lyrics + tags** on your own GPU workstation.

> **MVP0 Focus:** Stable local runs, clear outputs, easy debugging.  
> No hosting, no accounts, no business logic.

---

## Features (MVP0)

- **Local Gradio Web UI**: Simple, clean interface running on localhost.
- **Lyrics & Tags**: Input lyrics with sections (`[Verse]`, `[Chorus]`) and style tags.
- **Generation Controls**: Tuning for CFG, Temperature, Top-K, and Length.
- **Persistence**:
  - **SQLite DB**: Tracks all jobs, parameters, and results (`beatbunny.db`).
  - **Filesystem**: Audio and metadata saved to `outputs/job_<id>/`.
- **History**: Recall settings from previous runs instantly.
- **Artifacts**: Download WAV, MP3 (if ffmpeg available), and JSON metadata.

---

## Requirements

### Hardware
- **NVIDIA GPU** (Required for MVP0)
- **VRAM**: 8GB minimum recommended (12GB+ for longer generations).
- **Disk**: ~10GB for model weights + space for outputs.

### Software
- **Python 3.10** (Required - heartlib pins torch==2.4.1 which is not available for Python 3.11+)
- Git
- ffmpeg (Optional, for MP3 conversion)
- CUDA 12.4+ drivers (for GPU support)

---

## Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/beatbunny.git
cd beatbunny

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt

# IMPORTANT: After installing heartlib, you must reinstall PyTorch with CUDA support
# heartlib installs CPU-only torch by default
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
*Note: Use CUDA 12.4 index for torch 2.4.1. For other versions, visit [pytorch.org](https://pytorch.org).*

### 3. Download Model Weights
BeatBunny requires the **HeartMuLa-oss-3B** model weights. 
1. Download the weights from the upstream source (e.g., HuggingFace).
2. Place them in a folder, for example: `models/HeartMuLa-oss-3B`.

### 4. Configuration
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` to point to your model directory:
```ini
# .env
MODEL_DIR=./models/HeartMuLa-oss-3B
OUTPUT_DIR=./outputs
```

### 5. Run
```bash
python app.py
```
Open the link displayed in your terminal (usually `http://127.0.0.1:7860`).

---

## Troubleshooting

### "CUDA not available"
BeatBunny cannot find your GPU.
1. Run this check:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. If `False`, verify you have the CUDA-enabled torch:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   - Should show `2.4.1+cu124` (not `2.4.1+cpu`)
   - If CPU-only, reinstall: `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124`

### "CUDA Out of Memory"
The generation is too large for your VRAM.
- Reduce **Duration** (e.g., 60s → 30s).
- Reduce **CFG Scale**.
- Close other GPU-intensive applications (games, other models).

### Import Errors
If you see `ModuleNotFoundError: No module named 'heartmula'`, ensure you have the model's supporting libraries installed or included in your `PYTHONPATH`.

---

## License
MIT (Code) / Upstream License (Model Weights)
