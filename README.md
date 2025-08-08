# Ultra Fast Smart Whisper Server

High-performance audio transcription server optimized for RunPod.io with smart model management and configurable batch processing. [Docker Hub (~24GB and ~8GB compressed)](https://hub.docker.com/r/rushiy26/whisper-server)

## Evaluation

The server can process the audio file from the `test_input.json`, which is 1.5 hours long, in ~52 seconds with cold start at ~10 seconds using GPU CUDA 24GB with `batch_size=256`.

## Features

- **Single Model with Optimal Batch Processing**: Efficient VRAM utilization through large batch sizes
- **Optimized for Large Audio Files**: Superior performance for batch audio processing
- **Configurable Batch Size**: Environment variable for different GPU types
- **Fast Cold Starts**: Lazy loading with single model initialization
- **Simple and Reliable**: No complex model management overhead

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

### Cached model

The cached model and preprocessor should be located in the `/src/whisper_model_cache`. This folder is not included in this repo as it is ~1.5GB.
It can be created using `save_pipeline.py` file. Then if you will try to run the `handler.py` it should raise an error because of some value in the configuration (i don't remember which). You can then just search through the codebase to find that value and delete it in one of the main json files. Then you are free to go.

### RunPod Serverless Deployment

1. Build and push Docker image:
   ```bash
   docker build --platform linux/amd64 -t your-username/whisper-server:v1.0.0 .
   docker push your-username/whisper-server
   ```

2. Create RunPod endpoint with environment variables
3. Deploy and test with audio URL

### Example Request

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav"
  }
}
```

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

### Recommended Batch Sizes by GPU:

| GPU Type | VRAM | Recommended Batch Size |
|----------|------|----------------------|
| RTX 4000 | 16GB | 64 |
| RTX 4090 | 24GB | 128 |
| A100 | 40GB | 256 |
| H100 | 80GB | 512 |

Set via environment variable: `MODEL_BATCH_SIZE=128`

## Architecture

- **Single Model**: Simple, efficient single-model design
- **Lazy Loading**: Model loads on first request to minimize cold starts
- **Optimal Batching**: Large batch sizes maximize GPU utilization
- **Reliable**: No complex threading or model management