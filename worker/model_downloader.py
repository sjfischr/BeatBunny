"""
Model downloader for BeatBunny.
Downloads HeartMuLa models from HuggingFace using huggingface_hub.
"""

import os
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Generator
from pathlib import Path

logger = logging.getLogger(__name__)

# Model repository information
MODELS_TO_DOWNLOAD = [
    {
        "name": "HeartMuLaGen",
        "repo_id": "HeartMuLa/HeartMuLaGen",
        "local_dir_name": "",  # Download to root of model_dir
        "description": "Tokenizer and generation config (~9 MB)",
        "size_gb": 0.01,
    },
    {
        "name": "HeartMuLa-oss-3B",
        "repo_id": "HeartMuLa/HeartMuLa-oss-3B",
        "local_dir_name": "HeartMuLa-oss-3B",
        "description": "Main 3B parameter music generation model (~15.8 GB)",
        "size_gb": 15.8,
    },
    {
        "name": "HeartCodec-oss",
        "repo_id": "HeartMuLa/HeartCodec-oss",
        "local_dir_name": "HeartCodec-oss",
        "description": "Audio codec for encoding/decoding (~6.64 GB)",
        "size_gb": 6.64,
    },
]


@dataclass
class DownloadProgress:
    """Tracks download progress for UI updates."""
    model_name: str
    status: str = "pending"  # pending, downloading, completed, failed
    progress_percent: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    current_file: str = ""
    error_message: Optional[str] = None

    @property
    def status_emoji(self) -> str:
        if self.status == "completed":
            return "✅"
        elif self.status == "downloading":
            return "⏳"
        elif self.status == "failed":
            return "❌"
        return "⏸️"


@dataclass
class DownloadState:
    """Global download state for tracking across all models."""
    is_downloading: bool = False
    current_model: Optional[str] = None
    progress: dict = field(default_factory=dict)  # model_name -> DownloadProgress
    cancel_requested: bool = False
    
    def reset(self):
        self.is_downloading = False
        self.current_model = None
        self.progress = {}
        self.cancel_requested = False


# Global download state
_download_state = DownloadState()


def get_download_state() -> DownloadState:
    """Get the global download state."""
    return _download_state


def check_huggingface_hub() -> tuple[bool, str]:
    """Check if huggingface_hub is installed and working."""
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        return True, f"huggingface_hub v{version} installed"
    except ImportError:
        return False, "huggingface_hub not installed. Run: pip install huggingface_hub"


def download_model_sync(
    repo_id: str,
    local_dir: str,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> tuple[bool, str]:
    """
    Download a model from HuggingFace Hub synchronously.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "HeartMuLa/HeartMuLa-oss-3B")
        local_dir: Local directory to save the model
        progress_callback: Optional callback(status_message, progress_percent)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from huggingface_hub import snapshot_download
        
        os.makedirs(local_dir, exist_ok=True)
        
        if progress_callback:
            progress_callback(f"Starting download of {repo_id}...", 0)

        # Download the entire repository
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        if progress_callback:
            progress_callback(f"Completed: {repo_id}", 100)
            
        return True, f"Successfully downloaded {repo_id}"

    except Exception as e:
        error_msg = f"Failed to download {repo_id}: {str(e)}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(f"Error: {str(e)}", -1)
        return False, error_msg


def download_all_models_generator(model_dir: str) -> Generator[str, None, bool]:
    """
    Download all required models with progress updates.
    
    This is a generator that yields status messages for Gradio streaming.
    
    Args:
        model_dir: Base directory to store models
        
    Yields:
        Status messages for UI display
        
    Returns:
        True if all downloads succeeded, False otherwise
    """
    global _download_state
    
    # Check huggingface_hub first
    hf_ok, hf_msg = check_huggingface_hub()
    if not hf_ok:
        yield f"❌ {hf_msg}"
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        yield "❌ Failed to import huggingface_hub"
        return False

    _download_state.is_downloading = True
    _download_state.cancel_requested = False
    
    os.makedirs(model_dir, exist_ok=True)
    
    total_models = len(MODELS_TO_DOWNLOAD)
    all_success = True

    yield "## 🚀 Starting Model Download\n"
    yield f"**Target directory:** `{model_dir}`\n"
    yield f"**Total download size:** ~22.5 GB\n"
    yield "---\n"

    for idx, model_info in enumerate(MODELS_TO_DOWNLOAD, 1):
        if _download_state.cancel_requested:
            yield "\n⚠️ **Download cancelled by user.**"
            _download_state.is_downloading = False
            return False

        name = model_info["name"]
        repo_id = model_info["repo_id"]
        local_dir_name = model_info["local_dir_name"]
        description = model_info["description"]
        size_gb = model_info["size_gb"]

        _download_state.current_model = name
        
        if local_dir_name:
            local_dir = os.path.join(model_dir, local_dir_name)
        else:
            local_dir = model_dir

        yield f"\n### [{idx}/{total_models}] {name}\n"
        yield f"- **Description:** {description}\n"
        yield f"- **Repository:** `{repo_id}`\n"
        yield f"- **Size:** ~{size_gb} GB\n"
        yield f"- **Downloading to:** `{local_dir}`\n"
        yield "- **Status:** ⏳ Downloading...\n"

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            yield f"- **Status:** ✅ **Completed!**\n"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to download {name}: {error_msg}")
            yield f"- **Status:** ❌ **Failed:** {error_msg}\n"
            all_success = False
            # Continue with other models even if one fails

    _download_state.is_downloading = False
    _download_state.current_model = None

    yield "\n---\n"
    if all_success:
        yield "## ✅ All Models Downloaded Successfully!\n"
        yield "You can now use the **Generate** tab to create music.\n"
    else:
        yield "## ⚠️ Some Downloads Failed\n"
        yield "Please check the errors above and try again.\n"
        yield "You can re-run the download - it will resume where it left off.\n"

    return all_success


def download_single_model_generator(model_dir: str, model_name: str) -> Generator[str, None, bool]:
    """
    Download a single model component.
    
    Args:
        model_dir: Base directory to store models
        model_name: Name of the model to download (e.g., "HeartMuLa-oss-3B")
        
    Yields:
        Status messages for UI display
    """
    global _download_state
    
    # Find the model info
    model_info = None
    for m in MODELS_TO_DOWNLOAD:
        if m["name"] == model_name:
            model_info = m
            break
    
    if not model_info:
        yield f"❌ Unknown model: {model_name}"
        return False

    # Check huggingface_hub
    hf_ok, hf_msg = check_huggingface_hub()
    if not hf_ok:
        yield f"❌ {hf_msg}"
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        yield "❌ Failed to import huggingface_hub"
        return False

    _download_state.is_downloading = True
    _download_state.current_model = model_name

    repo_id = model_info["repo_id"]
    local_dir_name = model_info["local_dir_name"]
    
    if local_dir_name:
        local_dir = os.path.join(model_dir, local_dir_name)
    else:
        local_dir = model_dir

    os.makedirs(local_dir, exist_ok=True)

    yield f"## Downloading {model_name}\n"
    yield f"- **Repository:** `{repo_id}`\n"
    yield f"- **Size:** ~{model_info['size_gb']} GB\n"
    yield f"- **Destination:** `{local_dir}`\n"
    yield "- **Status:** ⏳ Downloading...\n"

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        yield f"\n✅ **{model_name} downloaded successfully!**\n"
        _download_state.is_downloading = False
        return True

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to download {model_name}: {error_msg}")
        yield f"\n❌ **Failed:** {error_msg}\n"
        _download_state.is_downloading = False
        return False


def cancel_download():
    """Request cancellation of the current download."""
    global _download_state
    _download_state.cancel_requested = True
    logger.info("Download cancellation requested")


def get_model_info_text() -> str:
    """Get formatted information about the models to be downloaded."""
    lines = [
        "## 📦 HeartMuLa Model Components",
        "",
        "BeatBunny requires three model components from HuggingFace:",
        "",
    ]
    
    total_size = 0
    for model in MODELS_TO_DOWNLOAD:
        total_size += model["size_gb"]
        lines.extend([
            f"### {model['name']}",
            f"- **HuggingFace:** [{model['repo_id']}](https://huggingface.co/{model['repo_id']})",
            f"- **Size:** ~{model['size_gb']} GB",
            f"- **Description:** {model['description']}",
            "",
        ])
    
    lines.extend([
        "---",
        f"**Total Download Size:** ~{total_size:.1f} GB",
        "",
        "**Requirements:**",
        "- Stable internet connection",
        "- Sufficient disk space (~25 GB recommended)",
        "- Download can be resumed if interrupted",
        "",
        "**Expected folder structure after download:**",
        "```",
        "models/",
        "├── gen_config.json",
        "├── tokenizer.json",
        "├── HeartMuLa-oss-3B/",
        "│   ├── config.json",
        "│   ├── model-00001-of-00004.safetensors",
        "│   ├── model-00002-of-00004.safetensors",
        "│   ├── model-00003-of-00004.safetensors",
        "│   ├── model-00004-of-00004.safetensors",
        "│   └── model.safetensors.index.json",
        "└── HeartCodec-oss/",
        "    ├── config.json",
        "    └── model.safetensors",
        "```",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    print(get_model_info_text())
    print("\n" + "="*50 + "\n")
    
    ok, msg = check_huggingface_hub()
    print(f"HuggingFace Hub: {msg}")
