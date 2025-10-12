import runpod
from runpod.serverless.utils import download_files_from_urls
import os
import subprocess

from model import load_whisper_pipeline

# Environment variables for configuration  
MODEL_BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '128'))

# Single model instance with lazy loading
model = None

unsupported_audio_file_extensions = ["m4a"]

def fix_metadata_fast(src, dst):
    print(f"🔄 Fixing metadata: {src} -> {dst}")
    subprocess.run([
        "ffmpeg", "-y", "-i",
        src,"-ar", "16000", "-ac",
        "1", "-c:a", "pcm_s16le", dst
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Metadata fixed: {src} -> {dst}")

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
    audio_file_extension = audio_file_path.split('.')[-1]
    audio_file_path_fixed = audio_file_path

    # Move the metadata to the beggining of the file for unsupported audio file extensions
    if audio_file_extension in unsupported_audio_file_extensions:
        audio_file_path_fixed = audio_file_path.replace(audio_file_extension, "wav")
        fix_metadata_fast(audio_file_path, audio_file_path_fixed)
    
    # Process with single optimized model
    model = get_model()
    result = model(audio_file_path_fixed)
    
    return result

runpod.serverless.start({"handler": handler})