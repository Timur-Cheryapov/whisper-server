import runpod
from runpod.serverless.utils import download_files_from_urls
import asyncio
import threading
import time
import os

from model import load_whisper_pipeline

# Environment variables for configuration
MAX_MODEL_INSTANCES = int(os.environ.get('MAX_MODEL_INSTANCES', '3'))
MODEL_BATCH_SIZE = int(os.environ.get('MODEL_BATCH_SIZE', '32'))

# Smart model management
class ModelManager:
    def __init__(self):
        self.models = []
        self.model_busy = [] # List of booleans indicating if a model is busy
        self.model_lock = threading.Lock()
        self.load_lock = threading.Lock()
        
    def get_available_model(self):
        """Get an available model or create a new one if needed"""
        with self.model_lock:
            # First, try to find an available model
            for i, busy in enumerate(self.model_busy):
                if not busy:
                    self.model_busy[i] = True # Mark the available model as busy
                    return i, self.models[i]
            
            # If all models are busy and we haven't reached the limit, create a new one
            if len(self.models) < MAX_MODEL_INSTANCES:
                print(f"🔄 Creating new model instance #{len(self.models) + 1}")
                with self.load_lock:  # Prevent multiple simultaneous model loads
                    new_model = load_whisper_pipeline("./whisper_model_cache", batch_size=MODEL_BATCH_SIZE)
                    self.models.append(new_model)
                    self.model_busy.append(True)
                    model_index = len(self.models) - 1
                    print(f"✅ Model instance #{model_index + 1} ready")
                    return model_index, new_model
            
            # If we've reached the limit, wait for the first available model
            print(f"⏳ All {MAX_MODEL_INSTANCES} models busy, waiting for availability...")
            return self._wait_for_available_model()
    
    def _wait_for_available_model(self):
        """Wait for any model to become available"""
        while True:
            with self.model_lock:
                for i, busy in enumerate(self.model_busy):
                    if not busy:
                        self.model_busy[i] = True
                        return i, self.models[i]
            time.sleep(1)  # Short sleep to avoid busy waiting
    
    def release_model(self, model_index):
        """Mark a model as available"""
        with self.model_lock:
            if model_index < len(self.model_busy):
                self.model_busy[model_index] = False

# Global model manager
model_manager = ModelManager()

def process_audio_with_smart_loading(audio_file_path):
    """Process audio with smart model management"""
    model_index, model = model_manager.get_available_model()
    try:
        result = model(audio_file_path)
        return result
    finally:
        model_manager.release_model(model_index)

async def handler(job):
    input_data = job['input']
    audio_url = input_data['audio_url']
    
    # Download file (keep your existing logic)
    audio_file_path = download_files_from_urls(job_id=job['id'], urls=audio_url)[0]
    
    # Process with smart model management
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Use default thread pool
        process_audio_with_smart_loading, 
        audio_file_path
    )
    
    return result

runpod.serverless.start({"handler": handler})