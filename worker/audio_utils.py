import os
import logging
import subprocess
import shutil

logger = logging.getLogger(__name__)


def convert_wav_to_mp3(wav_path, mp3_path=None):
    """
    Converts a WAV file to MP3 using ffmpeg if available.

    Args:
        wav_path (str): Path to input WAV file.
        mp3_path (str, optional): Path to output MP3 file.
                                  If None, uses same filename with .mp3 extension.

    Returns:
        str: Path to the generated MP3 file if successful, else None.
    """
    if not os.path.exists(wav_path):
        logger.warning(f"Audio conversion failed: Source file not found at {wav_path}")
        return None

    if mp3_path is None:
        mp3_path = os.path.splitext(wav_path)[0] + ".mp3"

    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        logger.info("ffmpeg not found on PATH. Skipping MP3 conversion.")
        return None

    try:
        # -y: Overwrite output files
        # -i: Input file
        # -q:a 0: Best variable bit rate quality
        # -map_metadata 0: Copy metadata
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            mp3_path,
        ]

        # Run subprocess quietly
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0 and os.path.exists(mp3_path):
            logger.info(f"Converted WAV to MP3: {mp3_path}")
            return mp3_path
        else:
            logger.warning(
                f"ffmpeg conversion failed with code {result.returncode}: {result.stderr}"
            )
            return None

    except Exception as e:
        logger.error(f"Error during MP3 conversion: {e}")
        return None
