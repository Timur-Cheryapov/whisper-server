FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /

# Install only essential packages in single layer
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ffmpeg \
        libgl1 \
        libx11-6 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies in optimized order
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --break-system-packages \
        torch --index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir --break-system-packages \
        -r /requirements.txt && \
    pip install --no-cache-dir --break-system-packages \
        huggingface_hub[hf_xet]

# Copy the rest of the application code
COPY src .

# Copy test input that will be used when the container runs outside of runpod
COPY test_input.json .

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]