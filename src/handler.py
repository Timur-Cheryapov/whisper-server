"""
RunPod serverless handler for Whisper transcription.
Handles audio file download, conversion, and transcription.
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import runpod
from runpod.serverless.utils import download_files_from_urls

from model import load_whisper_pipeline, CUDANotAvailableError

# Configure logging with structured format for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Environment variables for configuration
MODEL_BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", "128"))

# Single model instance with lazy loading
_model = None

# Audio formats that need conversion to WAV for Whisper compatibility
# These formats may have metadata issues or are not directly supported
FORMATS_REQUIRING_CONVERSION = frozenset({
    "m4a",   # AAC audio container - metadata issues
    "ogg",   # Ogg Vorbis - not well supported by some backends
    "opus",  # Opus codec - needs conversion
    "webm",  # WebM audio - container format issues
    "wma",   # Windows Media Audio
    "aac",   # Raw AAC
    "amr",   # Adaptive Multi-Rate (mobile recordings)
    "3gp",   # Mobile video/audio format
    "3gpp",  # Mobile video/audio format
})

# Formats known to work well with Whisper
SUPPORTED_FORMATS = frozenset({
    "wav", "mp3", "flac", "mp4", "mpeg", "mpga", "webm",
})


def get_job_logger(job_id: str) -> logging.LoggerAdapter:
    """Create a logger adapter that includes job ID in all messages."""
    return logging.LoggerAdapter(logger, {"job_id": job_id})


def get_file_extension(file_path: str) -> str:
    """
    Safely extract file extension from path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Lowercase file extension without the dot, or empty string if none
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    return ext[1:] if ext else ""


def convert_audio_to_wav(
    src_path: str,
    dst_path: str,
    job_id: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bool:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Uses CPU-based conversion which is fast enough for audio and doesn't
    require GPU resources. Outputs 16-bit PCM WAV at 16kHz mono for optimal
    Whisper compatibility.
    
    Args:
        src_path: Source audio file path
        dst_path: Destination WAV file path
        job_id: Job ID for logging
        sample_rate: Output sample rate (default 16000 for Whisper)
        channels: Output channels (default 1 for mono)
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    job_log = get_job_logger(job_id)
    job_log.info(f"Converting audio: {Path(src_path).name} -> {Path(dst_path).name}")
    
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",                    # Overwrite output file
                "-i", src_path,          # Input file
                "-vn",                   # No video
                "-ar", str(sample_rate), # Sample rate
                "-ac", str(channels),    # Audio channels
                "-c:a", "pcm_s16le",     # 16-bit PCM codec
                "-f", "wav",             # Force WAV format
                dst_path,
            ],
            capture_output=True,
            timeout=300,  # 5 minute timeout for long files
            check=False,
        )
        
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            job_log.error(f"ffmpeg conversion failed: {stderr[:500]}")
            return False
            
        job_log.info(f"Audio conversion completed: {Path(dst_path).name}")
        return True
        
    except subprocess.TimeoutExpired:
        job_log.error("Audio conversion timed out after 5 minutes")
        return False
    except FileNotFoundError:
        job_log.error("ffmpeg not found - ensure it's installed")
        return False
    except Exception as e:
        job_log.error(f"Audio conversion error: {e}")
        return False


@contextmanager
def cleanup_files(*paths: str):
    """
    Context manager to ensure temporary files are cleaned up.
    
    Args:
        *paths: File paths to clean up after the context exits
    """
    try:
        yield
    finally:
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass  # Best effort cleanup


def get_model():
    """
    Lazy load single model instance with optimal batch size.
    
    Returns:
        The loaded Whisper pipeline
        
    Raises:
        CUDANotAvailableError: If CUDA is not available
    """
    global _model
    if _model is None:
        logger.info(f"Initializing Whisper model with batch size: {MODEL_BATCH_SIZE}")
        _model = load_whisper_pipeline(
            "./whisper_model_cache",
            batch_size=MODEL_BATCH_SIZE,
            require_cuda=True,
        )
        logger.info("Model ready for processing")
    return _model


def validate_input(input_data: dict) -> tuple[bool, Optional[str]]:
    """
    Validate job input data.
    
    Args:
        input_data: The input dictionary from the job
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not input_data:
        return False, "Input data is empty"
    
    audio_url = input_data.get("audio_url")
    if not audio_url:
        return False, "Missing required field: audio_url"
    
    if not isinstance(audio_url, str):
        return False, "audio_url must be a string"
    
    return True, None


async def handler(job: dict) -> dict:
    """
    Main handler for RunPod serverless jobs.
    
    Processes audio transcription requests:
    1. Validates input
    2. Downloads audio file
    3. Converts to WAV if needed
    4. Transcribes using Whisper
    5. Returns transcription result
    
    Args:
        job: RunPod job dictionary containing 'id' and 'input'
        
    Returns:
        Transcription result or error dictionary
    """
    job_id = job.get("id", "unknown")
    job_log = get_job_logger(job_id)
    
    job_log.info("Job started")
    
    # Validate input
    input_data = job.get("input", {})
    is_valid, error_msg = validate_input(input_data)
    if not is_valid:
        job_log.error(f"Invalid input: {error_msg}")
        return {"error": error_msg}
    
    audio_url = input_data["audio_url"]
    job_log.info(f"Processing audio from URL: {audio_url[:100]}...")
    
    audio_file_path = None
    converted_file_path = None
    
    try:
        # Download audio file
        job_log.info("Downloading audio file...")
        downloaded_files = download_files_from_urls(job_id=job_id, urls=audio_url)
        
        if not downloaded_files:
            job_log.error("Failed to download audio file")
            return {"error": "Failed to download audio file"}
        
        audio_file_path = downloaded_files[0]
        job_log.info(f"Downloaded: {Path(audio_file_path).name}")
        
        # Determine if conversion is needed
        file_extension = get_file_extension(audio_file_path)
        processing_path = audio_file_path
        
        if file_extension in FORMATS_REQUIRING_CONVERSION:
            job_log.info(f"Format '{file_extension}' requires conversion to WAV")
            
            # Create converted file path
            converted_file_path = str(
                Path(audio_file_path).with_suffix(".converted.wav")
            )
            
            if not convert_audio_to_wav(
                audio_file_path, converted_file_path, job_id
            ):
                return {"error": f"Failed to convert {file_extension} to WAV"}
            
            processing_path = converted_file_path
        elif file_extension and file_extension not in SUPPORTED_FORMATS:
            job_log.warning(
                f"Unknown format '{file_extension}' - attempting direct processing"
            )
        
        # Load model and transcribe
        job_log.info("Starting transcription...")
        model = get_model()
        result = model(processing_path)
        
        job_log.info("Transcription completed successfully")
        return result
        
    except CUDANotAvailableError as e:
        job_log.error(f"CUDA error: {e}")
        return {"error": str(e)}
    except Exception as e:
        job_log.error(f"Transcription failed: {e}", exc_info=True)
        return {"error": f"Transcription failed: {str(e)}"}
    finally:
        # Cleanup temporary files
        files_to_cleanup = []
        if audio_file_path and os.path.exists(audio_file_path):
            files_to_cleanup.append(audio_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            files_to_cleanup.append(converted_file_path)
        
        for file_path in files_to_cleanup:
            try:
                os.remove(file_path)
                job_log.debug(f"Cleaned up: {Path(file_path).name}")
            except OSError:
                pass  # Best effort cleanup
        
        job_log.info("Job finished")


# Initialize RunPod serverless
runpod.serverless.start({"handler": handler})
