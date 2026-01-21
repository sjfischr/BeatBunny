"""
System status checking utilities for BeatBunny.
Checks CUDA availability, GPU info, and model readiness.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CudaStatus:
    """CUDA system status."""
    available: bool
    device_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def status_emoji(self) -> str:
        return "✅" if self.available else "❌"

    @property
    def summary(self) -> str:
        if self.available:
            return f"{self.device_name} ({self.vram_total_gb:.1f} GB VRAM)"
        return self.error_message or "CUDA not available"


@dataclass
class ModelComponentStatus:
    """Status of a single model component."""
    name: str
    repo_id: str
    local_path: str
    expected_size_gb: float
    required: bool = True
    
    # Status fields
    downloaded: bool = False
    files_found: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    size_on_disk_gb: float = 0.0

    @property
    def status_emoji(self) -> str:
        if self.downloaded:
            return "✅"
        elif self.files_found:
            return "⚠️"  # Partial download
        return "❌"

    @property
    def summary(self) -> str:
        if self.downloaded:
            return f"Ready ({self.size_on_disk_gb:.2f} GB)"
        elif self.files_found:
            return f"Incomplete ({len(self.files_found)}/{len(self.files_found) + len(self.files_missing)} files)"
        return "Not downloaded"


@dataclass
class ModelStatus:
    """Overall model status."""
    components: Dict[str, ModelComponentStatus]
    model_dir: str
    
    @property
    def all_ready(self) -> bool:
        return all(c.downloaded for c in self.components.values() if c.required)

    @property
    def status_emoji(self) -> str:
        return "✅" if self.all_ready else "❌"

    @property
    def total_size_gb(self) -> float:
        return sum(c.expected_size_gb for c in self.components.values())

    @property
    def downloaded_size_gb(self) -> float:
        return sum(c.size_on_disk_gb for c in self.components.values())


# Expected files for each model component
MODEL_COMPONENTS = {
    "HeartMuLaGen": {
        "repo_id": "HeartMuLa/HeartMuLaGen",
        "expected_size_gb": 0.01,  # ~9 MB
        "required": True,
        "expected_files": ["gen_config.json", "tokenizer.json"],
        "subfolder": "",  # Root of model_dir
    },
    "HeartMuLa-oss-3B": {
        "repo_id": "HeartMuLa/HeartMuLa-oss-3B",
        "expected_size_gb": 15.8,
        "required": True,
        "expected_files": [
            "config.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
        ],
        "subfolder": "HeartMuLa-oss-3B",
    },
    "HeartCodec-oss": {
        "repo_id": "HeartMuLa/HeartCodec-oss",
        "expected_size_gb": 6.64,
        "required": True,
        "expected_files": ["config.json", "model.safetensors"],
        "subfolder": "HeartCodec-oss",
    },
}


def check_cuda_status() -> CudaStatus:
    """
    Check CUDA availability and GPU information.
    Returns CudaStatus with all relevant info.
    """
    try:
        import torch
    except ImportError:
        return CudaStatus(
            available=False,
            error_message="PyTorch not installed. Run: pip install torch"
        )

    if not torch.cuda.is_available():
        return CudaStatus(
            available=False,
            error_message="CUDA not available. Install CUDA-enabled PyTorch from pytorch.org"
        )

    try:
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024**3)
        
        # Get free memory
        vram_free = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        
        # Get CUDA version
        cuda_version = torch.version.cuda

        return CudaStatus(
            available=True,
            device_name=device_name,
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            cuda_version=cuda_version,
        )

    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return CudaStatus(
            available=False,
            error_message=f"Error checking CUDA: {str(e)}"
        )


def get_folder_size_gb(folder_path: str) -> float:
    """Calculate total size of a folder in GB."""
    total_size = 0
    folder = Path(folder_path)
    if folder.exists():
        for file in folder.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
    return total_size / (1024**3)


def check_model_component(model_dir: str, component_name: str, component_info: dict) -> ModelComponentStatus:
    """Check status of a single model component."""
    subfolder = component_info["subfolder"]
    
    if subfolder:
        local_path = os.path.join(model_dir, subfolder)
    else:
        local_path = model_dir

    status = ModelComponentStatus(
        name=component_name,
        repo_id=component_info["repo_id"],
        local_path=local_path,
        expected_size_gb=component_info["expected_size_gb"],
        required=component_info["required"],
    )

    expected_files = component_info["expected_files"]
    
    for filename in expected_files:
        filepath = os.path.join(local_path, filename)
        if os.path.exists(filepath):
            status.files_found.append(filename)
        else:
            status.files_missing.append(filename)

    status.downloaded = len(status.files_missing) == 0 and len(status.files_found) > 0
    status.size_on_disk_gb = get_folder_size_gb(local_path) if subfolder else 0

    # For root files (HeartMuLaGen), calculate size individually
    if not subfolder:
        for filename in status.files_found:
            filepath = os.path.join(local_path, filename)
            if os.path.exists(filepath):
                status.size_on_disk_gb += os.path.getsize(filepath) / (1024**3)

    return status


def check_model_status(model_dir: str) -> ModelStatus:
    """
    Check the status of all required model components.
    
    Args:
        model_dir: Path to the models directory
        
    Returns:
        ModelStatus with status of each component
    """
    components = {}
    
    for name, info in MODEL_COMPONENTS.items():
        components[name] = check_model_component(model_dir, name, info)

    return ModelStatus(components=components, model_dir=model_dir)


def get_system_status_text(model_dir: str) -> str:
    """
    Get a formatted text summary of system status.
    Useful for displaying in the UI.
    """
    cuda = check_cuda_status()
    models = check_model_status(model_dir)
    
    lines = [
        "## 🖥️ System Status",
        "",
        "### GPU (CUDA)",
        f"{cuda.status_emoji} **Status:** {'Available' if cuda.available else 'Not Available'}",
    ]
    
    if cuda.available:
        lines.extend([
            f"- **Device:** {cuda.device_name}",
            f"- **VRAM:** {cuda.vram_total_gb:.1f} GB total, {cuda.vram_free_gb:.1f} GB free",
            f"- **CUDA Version:** {cuda.cuda_version}",
        ])
    else:
        lines.append(f"- **Error:** {cuda.error_message}")

    lines.extend([
        "",
        "### Models",
        f"{models.status_emoji} **Overall:** {'All models ready!' if models.all_ready else 'Models need to be downloaded'}",
        f"- **Location:** `{models.model_dir}`",
        f"- **Total Required:** ~{models.total_size_gb:.1f} GB",
        f"- **Downloaded:** {models.downloaded_size_gb:.2f} GB",
        "",
    ])

    for name, comp in models.components.items():
        req_label = "(Required)" if comp.required else "(Optional)"
        lines.append(f"#### {comp.status_emoji} {name} {req_label}")
        lines.append(f"- **Status:** {comp.summary}")
        lines.append(f"- **HuggingFace:** `{comp.repo_id}`")
        if comp.files_missing:
            lines.append(f"- **Missing:** {', '.join(comp.files_missing[:3])}{'...' if len(comp.files_missing) > 3 else ''}")
        lines.append("")

    return "\n".join(lines)


def is_ready_to_generate(model_dir: str) -> tuple[bool, str]:
    """
    Quick check if the system is ready for generation.
    
    Returns:
        Tuple of (is_ready, message)
    """
    cuda = check_cuda_status()
    if not cuda.available:
        return False, f"❌ CUDA not available: {cuda.error_message}"

    models = check_model_status(model_dir)
    if not models.all_ready:
        missing = [name for name, c in models.components.items() if not c.downloaded and c.required]
        return False, f"❌ Missing models: {', '.join(missing)}. Go to Setup tab to download."

    return True, "✅ System ready for generation!"


if __name__ == "__main__":
    # Test the module
    print("Testing system status check...\n")
    print(get_system_status_text("./models"))
