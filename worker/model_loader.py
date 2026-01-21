import os
import logging
import torch

logger = logging.getLogger(__name__)

# Module-level cache for loaded models
_LOADED_MODELS = None


def validate_cuda():
    """
    Checks for CUDA availability and logs device info.
    Raises RuntimeError if CUDA is not available.
    """
    if not torch.cuda.is_available():
        error_msg = (
            "CUDA is not available. BeatBunny requires an NVIDIA GPU with CUDA support.\n"
            "Please ensure NVIDIA drivers and a CUDA-enabled PyTorch version are installed.\n"
            "See README.md for troubleshooting."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    device_name = torch.cuda.get_device_name(0)

    # Best-effort VRAM logging
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_info = f"{total_memory:.2f} GB"
    except Exception:
        vram_info = "Unknown VRAM"

    logger.info(f"CUDA Available: {device_name} | VRAM: {vram_info}")
    return True


def get_models(model_dir):
    """
    Loads HeartMuLa-oss-3B and HeartCodec from model_dir.
    Returns cached instance if already loaded.
    """
    global _LOADED_MODELS

    if _LOADED_MODELS is not None:
        logger.info("Using cached models.")
        return _LOADED_MODELS

    validate_cuda()

    logger.info(f"Loading models from: {model_dir}")

    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    try:
        from heartlib import HeartMuLaGenPipeline
        
        logger.info("Initializing HeartMuLa model pipeline...")
        
        # Load the HeartMuLa generation pipeline
        # This loads both HeartMuLa-oss-3B and HeartCodec-oss
        pipeline = HeartMuLaGenPipeline.from_pretrained(
            model_dir,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            version="3B"
        )
        
        _LOADED_MODELS = {
            "pipeline": pipeline,
            "device": "cuda",
        }

        logger.info("Models loaded successfully.")

    except ImportError as e:
        logger.error("Failed to import HeartMuLa library.")
        logger.error("Ensure heartlib is installed: pip install git+https://github.com/HeartMuLa/heartlib.git")
        raise e
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

    return _LOADED_MODELS
