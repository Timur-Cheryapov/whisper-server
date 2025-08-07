FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Update and upgrade the system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python (using default Python 3.12 from Ubuntu 24.04)
RUN apt-get update -y && \
    apt-get install python3 python3-dev python3-venv python3-pip -y --no-install-recommends && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install huggingface_hub[hf_xet] --break-system-packages && \
    pip install torch --index-url https://download.pytorch.org/whl/cu129 --break-system-packages && \
    pip install -r /requirements.txt --no-cache-dir --break-system-packages

# Copy the rest of the application code
COPY src .

# Copy test input that will be used when the container runs outside of runpod
COPY test_input.json .

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]