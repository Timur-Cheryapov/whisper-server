"""
Script to download and save the Whisper pipeline locally for offline use in Docker.
Run this script once to download all model weights and save them locally.
"""

import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pickle

def save_whisper_pipeline(save_dir="./whisper_model_cache"):
    """
    Download and save the Whisper pipeline components locally.
    
    Args:
        save_dir (str): Directory to save the model components
    """
    print("🚀 Starting pipeline download and save process...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Device configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"💻 Using device: {device}")
    print(f"🔢 Using dtype: {torch_dtype}")
    
    model_id = "openai/whisper-large-v3-turbo"
    
    print(f"📥 Downloading model: {model_id}")
    
    # Download and save model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True,
        cache_dir=os.path.join(save_dir, "model_cache")
    )
    
    # Save model locally
    model_save_path = os.path.join(save_dir, "model")
    model.save_pretrained(model_save_path)
    print(f"✅ Model saved to: {model_save_path}")
    
    # Download and save processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=os.path.join(save_dir, "processor_cache")
    )
    
    processor_save_path = os.path.join(save_dir, "processor")
    processor.save_pretrained(processor_save_path)
    print(f"✅ Processor saved to: {processor_save_path}")
    
    # Save pipeline configuration
    pipeline_config = {
        "model_id": model_id,
        "task": "automatic-speech-recognition",
        "chunk_length_s": 30,
        "batch_size": 32,
        "torch_dtype": torch_dtype,
        "device": device
    }
    
    config_path = os.path.join(save_dir, "pipeline_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(pipeline_config, f)
    print(f"✅ Pipeline config saved to: {config_path}")
    
    print(f"🎉 Pipeline successfully saved to: {save_dir}")
    print(f"📁 Total size: {get_dir_size(save_dir):.2f} MB")
    
    return save_dir

def get_dir_size(path):
    """Calculate directory size in MB"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total / (1024 * 1024)  # Convert to MB
    except OSError:
        return 0

if __name__ == "__main__":
    save_dir = save_whisper_pipeline()
    print(f"\n🐳 To use in Docker, copy the '{save_dir}' directory to your Docker container")
    print("📋 Then use load_pipeline.py to load the saved components")