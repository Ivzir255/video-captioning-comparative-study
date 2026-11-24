# ─────────────────────────────────────────────────────────────────────────────
# Video Captioning Project — GPU-Enabled Docker Image
# Base: CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
# Instructor: Dr. Akhtar Jamil | Generative AI, FAST NUCES, Spring 2026
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="GenAI Project Team"
LABEL description="Comparative Video Captioning: CLIP+GPT-2, ViViT, BLIP-2, CLIP4Caption, LLaVA-NeXT, Video-LLaMA, Vid2Seq"

# ── System packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3.10 python3.10-dev python3-pip \
        git wget curl vim \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
        libopencv-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3        1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118 \
 && pip install --no-cache-dir -r requirements.txt

# ── Install CLIP (not on PyPI) ────────────────────────────────────────────────
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# ── Copy project code ─────────────────────────────────────────────────────────
COPY shared/  ./shared/
COPY *.ipynb  ./
COPY *.py     ./

# ── MLflow tracking server port ───────────────────────────────────────────────
EXPOSE 5000

# ── Jupyter port ─────────────────────────────────────────────────────────────
EXPOSE 8888

# ── Create directories ────────────────────────────────────────────────────────
RUN mkdir -p ./videos/msvd ./videos/msrvtt ./results/figures ./results/tables \
             ./mlruns

# ── Entry point: start Jupyter Lab ───────────────────────────────────────────
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
