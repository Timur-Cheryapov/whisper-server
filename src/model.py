"""
Script to load the saved Whisper pipeline from local files (offline).
Use this in Docker containers where internet access is limited.
"""

import os
import torch
import pickle
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def load_whisper_pipeline(save_dir="./whisper_model_cache", batch_size=128):
    """
    Load the Whisper pipeline from locally saved components.
    
    Args:
        save_dir (str): Directory containing the saved model components
        batch_size (int): Batch size for the pipeline (default: 32)
        
    Returns:
        pipeline: The loaded speech recognition pipeline
    """
    print("🔄 Loading Whisper pipeline from local files...")
    
    # Check if save directory exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"❌ Save directory not found: {save_dir}")
    
    # Load pipeline configuration
    config_path = os.path.join(save_dir, "pipeline_config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Pipeline config not found: {config_path}")
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    print(f"📋 Loaded config for model: {config['model_id']}")
    
    # Device configuration (prefer saved config but adapt to current environment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"💻 Using device: {device}")
    print(f"🔢 Using dtype: {torch_dtype}")
    
    # Load model from local files
    model_path = os.path.join(save_dir, "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model directory not found: {model_path}")
    
    print("📥 Loading model from local files...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,  # Ensure no internet access
    )
    model.to(device)
    print("✅ Model loaded successfully")
    
    # Load processor from local files
    processor_path = os.path.join(save_dir, "processor")
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"❌ Processor directory not found: {processor_path}")
    
    print("📥 Loading processor from local files...")
    processor = AutoProcessor.from_pretrained(
        processor_path,
        local_files_only=True,  # Ensure no internet access
    )
    print("✅ Processor loaded successfully")
    
    # Create pipeline
    print(f"🔧 Creating pipeline with batch size: {batch_size}")
    pipe = pipeline(
        config["task"],
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=config.get("chunk_length_s", 30),
        batch_size=batch_size,  # Use configurable batch size
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("🎉 Pipeline loaded successfully from local files!")
    return pipe