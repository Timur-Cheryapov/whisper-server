import runpod
from runpod.serverless.utils import download_files_from_urls
import asyncio
from concurrent.futures import ThreadPoolExecutor

from model import load_whisper_pipeline

# Load multiple model instances
NUM_INSTANCES = 3  # Adjust based on VRAM
models = [load_whisper_pipeline("./whisper_model_cache") for _ in range(NUM_INSTANCES)]
executor = ThreadPoolExecutor(max_workers=NUM_INSTANCES)

def process_audio_threaded(audio_file_path, model_index):
    """Process audio with specific model instance"""
    return models[model_index](audio_file_path)

async def handler(job):
    input_data = job['input']
    audio_url = input_data['audio_url']
    
    # Download file (keep your existing logic)
    audio_file_path = download_files_from_urls(job_id=job['id'], urls=audio_url)[0]
    
    # Process with available model instance
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        process_audio_threaded, 
        audio_file_path,
        hash(job['id']) % NUM_INSTANCES  # Simple load balancing
    )
    
    return result

runpod.serverless.start({"handler": handler})