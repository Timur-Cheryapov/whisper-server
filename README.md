# Ultra Fast Smart Whisper Server

High-performance audio transcription server optimized for RunPod.io with smart model management and configurable batch processing. [Docker Hub (~24GB and ~8GB compressed)](https://hub.docker.com/r/rushiy26/whisper-server)

## Evaluation

The server can process the audio file from the `test_input.json`, which is 1.5 hours long, in ~52 seconds with cold start at ~10 seconds using GPU CUDA 24GB with `batch_size=256`.

## Features

- **Single Model with Optimal Batch Processing**: Efficient VRAM utilization through large batch sizes
- **Optimized for Large Audio Files**: Superior performance for batch audio processing
- **Configurable Batch Size**: Environment variable for different GPU types
- **Fast Cold Starts**: Lazy loading with single model initialization
- **Automatic Audio Conversion**: Handles m4a, ogg, opus, webm, and other formats automatically
- **Production-Ready Logging**: Structured logs with job ID tracking
- **Robust Error Handling**: Proper cleanup and meaningful error messages

## Quick Start

### Environment Variables

Configure the batch size for your GPU type:

```bash
# For RTX 4090 (24GB VRAM)
MODEL_BATCH_SIZE=256

# For RTX 4000 (16GB VRAM) 
MODEL_BATCH_SIZE=128

# For A100 (40GB VRAM)
MODEL_BATCH_SIZE=512
```

### Cached Model Setup

The cached model and preprocessor should be located in `/whisper_model_cache` in the Docker container (or `./whisper_model_cache` relative to the handler). This folder is not included in this repo as it is ~1.5GB.

**To create the cached model:**

1. Run the save script:
   ```bash
   python save_pipeline.py
   ```

2. The script will download and save:
   - Model weights to `./whisper_model_cache/model/`
   - Processor to `./whisper_model_cache/processor/`
   - Pipeline config to `./whisper_model_cache/pipeline_config.pkl`

3. Copy the `whisper_model_cache` folder to your Docker build context before building.

**Troubleshooting:** If you encounter config-related errors when loading, check `pipeline_config.pkl` and remove any device-specific values like `torch_dtype` or `device` that might conflict with runtime detection.

### RunPod Serverless Deployment

1. Build and push Docker image:
```bash
docker build --platform linux/amd64 -t rushiy26/whisper-server:v1.0.0 .
docker push rushiy26/whisper-server:v1.0.0
```

2. Create RunPod endpoint with environment variables:
  - `MODEL_BATCH_SIZE`: Set based on your GPU (see table below)

3. Deploy and test with audio URL

### Example Request

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav"
  }
}
```

### Supported Audio Formats

**Directly Supported:**
- WAV, MP3, FLAC, MP4, MPEG, MPGA

**Auto-Converted (via ffmpeg):**
- M4A, OGG, OPUS, WEBM, WMA, AAC, AMR, 3GP, 3GPP

Files in the auto-convert list are converted to 16kHz mono WAV before transcription.

## Performance

### Cold Start Optimization
- **Fast Initialization**: Single model loads in 3-4 seconds
- **Lazy Loading**: Model loads only when first request arrives
- **No Overhead**: Simple, reliable single-model architecture

### Batch Processing
- **High Efficiency**: Large configurable batch sizes for optimal VRAM usage
- **GPU Optimized**: Batch size recommendations for different GPU types
- **Superior Performance**: Outperforms standard implementations for large audio files

## Configuration

### Recommended Batch Sizes by GPU

| GPU Type | VRAM | Recommended Batch Size |
|----------|------|------------------------|
| RTX 4000 | 16GB | 128                    |
| RTX 4090 | 24GB | 256                    |
| A100     | 40GB | 512                    |
| H100     | 80GB | 1024                   |

Set via environment variable: `MODEL_BATCH_SIZE=128`

## Architecture

- **Single Model**: Simple, efficient single-model design
- **Lazy Loading**: Model loads on first request to minimize cold starts
- **Optimal Batching**: Large batch sizes maximize GPU utilization
- **Reliable**: No complex threading or model management

## Error Handling

The server provides clear error messages for common issues:

- **CUDA Not Available**: Detailed message about CUDA/driver status
- **Unsupported Format**: Automatic conversion for known problematic formats
- **Download Failures**: Clear indication of network issues
- **Transcription Errors**: Logged with full context for debugging

## Logging

Logs are structured for production use:

```
2025-01-18 12:00:00 | INFO | Job started
2025-01-18 12:00:01 | INFO | Processing audio from URL: https://...
2025-01-18 12:00:02 | INFO | Downloaded: audio.m4a
2025-01-18 12:00:02 | INFO | Format 'm4a' requires conversion to WAV
2025-01-18 12:00:05 | INFO | Audio conversion completed: audio.converted.wav
2025-01-18 12:00:05 | INFO | Starting transcription...
2025-01-18 12:00:57 | INFO | Transcription completed successfully
2025-01-18 12:00:57 | INFO | Job finished
```

## Local Testing

### 1. Inspect/Fix the Pipeline Config

The `pipeline_config.pkl` is a binary file. Use the inspect script to view or fix it:

```bash
# View current config
python scripts/inspect_config.py

# Remove problematic keys (torch_dtype, device, batch_size)
python scripts/inspect_config.py --fix
```

### 2. Test Docker Container

**Build the image:**
```bash
docker build --platform linux/amd64 -t whisper-server:test .
```

**Test with RunPod's local test mode:**
```bash
# RunPod automatically uses test_input.json when not running on RunPod infrastructure
docker run --gpus all -it --rm \
  -e MODEL_BATCH_SIZE=32 \
  whisper-server:test
```

**Test with custom input:**
```bash
# Create a custom test input
echo '{"input": {"audio_url": "https://your-url.com/audio.mp3"}}' > custom_test.json

# Run with custom input mounted
docker run --gpus all -it --rm \
  -e MODEL_BATCH_SIZE=128 \
  -v $(pwd)/custom_test.json:/test_input.json \
  whisper-server:test
```

**Interactive debugging:**
```bash
# Start container with bash for debugging
docker run --gpus all -it --rm \
  -e MODEL_BATCH_SIZE=128 \
  whisper-server:test \
  bash

# Inside container, test CUDA:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run handler manually:
python -u /handler.py
```

## Dependencies

- **transformers**: >=4.57.0
- **runpod**: >=1.8.1
- **torch**: CUDA 12.9 compatible
- **ffmpeg**: For audio conversion (included in Docker image)
- **accelerate**: For optimized model loading
