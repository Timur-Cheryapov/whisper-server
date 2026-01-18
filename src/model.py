"""
Script to load the saved Whisper pipeline from local files (offline).
Use this in Docker containers where internet access is limited.
"""

import os
import warnings
import logging
import pickle

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Suppress specific transformers warnings about logits processors and deprecated inputs
warnings.filterwarnings("ignore", message=".*SuppressTokensAtBeginLogitsProcessor.*")
warnings.filterwarnings("ignore", message=".*SuppressTokensLogitsProcessor.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*generation_config.*default values have been modified.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*input name.*inputs.*is deprecated.*")

logger = logging.getLogger(__name__)


class CUDANotAvailableError(Exception):
    """Raised when CUDA is required but not available."""
    pass


def verify_cuda_availability(require_cuda: bool = True) -> tuple[str, torch.dtype]:
    """
    Verify CUDA availability and return device configuration.
    
    Args:
        require_cuda: If True, raises error when CUDA is not available
        
    Returns:
        Tuple of (device_string, torch_dtype)
        
    Raises:
        CUDANotAvailableError: When CUDA is required but not available
    """
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        # Verify CUDA is actually working by trying to allocate a small tensor
        try:
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            
            device = "cuda"
            torch_dtype = torch.float16
            
            # Log CUDA device info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA verified: {gpu_name} ({gpu_memory:.1f}GB)")
            
        except RuntimeError as e:
            if require_cuda:
                raise CUDANotAvailableError(
                    f"CUDA reported as available but failed to initialize: {e}"
                )
            logger.warning(f"CUDA initialization failed, falling back to CPU: {e}")
            device = "cpu"
            torch_dtype = torch.float32
    else:
        if require_cuda:
            raise CUDANotAvailableError(
                "CUDA is not available. Ensure NVIDIA drivers and CUDA toolkit are properly installed. "
                "Check: nvidia-smi, torch.cuda.is_available()"
            )
        logger.warning("CUDA not available, using CPU (this will be slow)")
        device = "cpu"
        torch_dtype = torch.float32
    
    return device, torch_dtype


def load_whisper_pipeline(
    save_dir: str = "./whisper_model_cache",
    batch_size: int = 128,
    require_cuda: bool = True
) -> pipeline:
    """
    Load the Whisper pipeline from locally saved components.
    
    Args:
        save_dir: Directory containing the saved model components
        batch_size: Batch size for the pipeline
        require_cuda: If True, raises error when CUDA is not available
        
    Returns:
        pipeline: The loaded speech recognition pipeline
        
    Raises:
        FileNotFoundError: When required files are missing
        CUDANotAvailableError: When CUDA is required but not available
    """
    logger.info("Loading Whisper pipeline from local files...")
    
    # Validate save directory exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Save directory not found: {save_dir}")
    
    # Load pipeline configuration
    config_path = os.path.join(save_dir, "pipeline_config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    logger.info(f"Loaded config for model: {config['model_id']}")
    
    # Verify CUDA and get device configuration
    device, torch_dtype = verify_cuda_availability(require_cuda=require_cuda)
    logger.info(f"Device: {device}, dtype: {torch_dtype}")
    
    # Load model from local files
    model_path = os.path.join(save_dir, "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    logger.info("Loading model from local files...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    logger.info("Model loaded successfully")
    
    # Load processor from local files
    processor_path = os.path.join(save_dir, "processor")
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Processor directory not found: {processor_path}")
    
    logger.info("Loading processor from local files...")
    processor = AutoProcessor.from_pretrained(
        processor_path,
        local_files_only=True,
    )
    logger.info("Processor loaded successfully")
    
    # Create pipeline with optimized settings
    # Note: We avoid passing conflicting generation parameters to prevent warnings
    logger.info(f"Creating pipeline with batch size: {batch_size}")
    pipe = pipeline(
        config["task"],
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=config.get("chunk_length_s", 30),
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device=device,
        # Use input_features instead of deprecated inputs parameter
        return_timestamps=False,
    )
    
    logger.info("Pipeline loaded successfully from local files")
    return pipe
