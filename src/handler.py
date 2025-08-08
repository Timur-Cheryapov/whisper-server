import runpod
from runpod.serverless.utils import download_files_from_urls
import os

from model import load_whisper_pipeline

# Environment variables for configuration  
MODEL_BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '128'))

# Single model instance with lazy loading
model = None

def get_model():
    """Lazy load single model instance with optimal batch size"""
    global model
    if model is None:
        print(f"🔄 Initializing Whisper model with batch size: {MODEL_BATCH_SIZE}")
        model = load_whisper_pipeline("./whisper_model_cache", batch_size=MODEL_BATCH_SIZE)
        print(f"✅ Model ready for high-efficiency batch processing")
    return model

async def handler(job):
    input_data = job['input']
    audio_url = input_data['audio_url']
    
    # Download file
    audio_file_path = download_files_from_urls(job_id=job['id'], urls=audio_url)[0]
    
    # Process with single optimized model
    model = get_model()
    result = model(audio_file_path)
    
    return result

runpod.serverless.start({"handler": handler})