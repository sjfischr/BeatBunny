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

    # TODO: Replace with actual HeartMuLa imports once upstream package is available
    # For MVP0, we assume the user has the code available or we use a placeholder structure
    # This block mimics the loading process

    try:
        # Example: from heartmula import HeartMuLa
        # Example: from heartcodec import HeartCodec

        # Placeholder for actual loading logic
        # model = HeartMuLa.from_pretrained(model_dir)
        # codec = HeartCodec.from_pretrained(model_dir)

        # Simulating a load for the scaffold
        logger.info("Initializing HeartMuLa model (Placeholder)...")

        # Real implementation would look like:
        # model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).cuda()
        # codec = AudioCodec.from_pretrained(model_dir).cuda()

        # For now, we return a stub dictionary or object
        _LOADED_MODELS = {
            "model": "Placeholder_HeartMuLa_3B",
            "codec": "Placeholder_HeartCodec",
            "device": "cuda",
        }

        logger.info("Models loaded successfully.")

    except ImportError as e:
        logger.error("Failed to import model libraries.")
        logger.error("Ensure all dependencies are installed per README.")
        raise e
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

    return _LOADED_MODELS
