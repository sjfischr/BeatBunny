import os
import json
import logging
import time
import traceback
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Import db module properly - assuming running from root
from db.db import update_job_status, add_artifact
from worker.model_loader import get_models
from worker.audio_utils import convert_wav_to_mp3

logger = logging.getLogger(__name__)


def run_generation_job(job_id, lyrics_text, tags_text, params, config):
    """
    Executes a music generation job.

    Args:
        job_id (str): UUID of the job
        lyrics_text (str): Input lyrics
        tags_text (str): Input style tags
        params (dict): Generation parameters (cfg, temp, top_k, length, seed)
        config (class): App configuration class containing directory paths

    Returns:
        dict: Result dictionary with status and output path
    """
    start_time = time.time()
    job_dir = os.path.join(config.OUTPUT_DIR, f"job_{job_id}")

    try:
        # 1. Update status to running
        logger.info(f"Starting job {job_id}")
        update_job_status(job_id, "processing")

        # Create job directory
        os.makedirs(job_dir, exist_ok=True)

        # 2. Persist inputs for reproducibility
        with open(os.path.join(job_dir, "lyrics.txt"), "w", encoding="utf-8") as f:
            f.write(lyrics_text)

        with open(os.path.join(job_dir, "tags.txt"), "w", encoding="utf-8") as f:
            f.write(tags_text)

        # 3. Load Models
        logger.info(f"Job {job_id}: Loading models...")
        models = get_models(config.MODEL_DIR)

        # 4. Set random seed for reproducibility
        seed = params.get("seed")
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Job {job_id}: Using seed {seed}")

        # 5. Generate Audio (Placeholder Logic)
        logger.info(f"Job {job_id}: Generating audio with params: {params}")

        # Extract params
        duration_s = int(params.get("max_length", 30))
        sample_rate = 32000  # Standard for many models, adjust as needed

        # --- MOCK GENERATION START ---
        # Replace this block with actual HeartMuLa inference:
        # audio_tensor = models['model'].generate(lyrics, tags, **params)

        # Simulating generation time
        time.sleep(2)

        # Generate silence/noise for MVP testing if no real model
        # 0.5s of white noise so we have a valid WAV file
        audio_data = np.random.uniform(-0.1, 0.1, int(sample_rate * duration_s))
        # --- MOCK GENERATION END ---

        # 6. Save Outputs
        wav_filename = "audio.wav"
        wav_path = os.path.join(job_dir, wav_filename)

        sf.write(wav_path, audio_data, sample_rate)
        logger.info(f"Job {job_id}: Saved audio to {wav_path}")

        # Save metadata.json
        runtime = time.time() - start_time
        metadata = {
            "job_id": job_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": round(runtime, 2),
            "params": params,
            "model_version": "HeartMuLa-oss-3B-MVP0",
            "lyrics_snippet": lyrics_text[:50] + "..."
            if len(lyrics_text) > 50
            else lyrics_text,
            "tags": tags_text,
        }

        metadata_path = os.path.join(job_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 7. Update DB Artifacts
        add_artifact(job_id, "audio_wav", wav_path)
        add_artifact(job_id, "metadata_json", metadata_path)

        # Optional: Convert to MP3
        mp3_path = convert_wav_to_mp3(wav_path)
        if mp3_path:
            add_artifact(job_id, "audio_mp3", mp3_path)

        # 8. Mark Succeeded
        update_job_status(
            job_id,
            "completed",
            {
                "status": "completed"
                # Could add runtime column to DB if schema allows, currently schema is minimal
            },
        )

        logger.info(f"Job {job_id}: Completed successfully in {runtime:.2f}s")
        return {"success": True, "job_id": job_id, "audio_path": wav_path}

    except RuntimeError as e:
        error_str = str(e)
        if "CUDA out of memory" in error_str:
            friendly_msg = (
                "GPU Out of Memory. Try: "
                "1) Reducing max length (e.g., to 30s), "
                "2) Lowering CFG scale, "
                "3) Closing other GPU apps."
            )
            logger.error(f"Job {job_id} OOM: {friendly_msg}")
            update_job_status(job_id, "failed", {"error": friendly_msg})
            return {"success": False, "error": friendly_msg}
        else:
            raise e

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        logger.error(traceback.format_exc())

        update_job_status(job_id, "failed", {"error": str(e)})
        return {"success": False, "error": str(e)}
