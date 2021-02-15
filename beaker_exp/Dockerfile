FROM python:3.8.5-buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir transformers -q
RUN pip install --no-cache-dir wandb -q
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en

COPY models/ models/
#COPY wandb/ wandb/
COPY mosaic/ mosaic/
COPY comet_t5.py comet_t5.py
COPY gpt2-zeroshot.py gpt2-zeroshot.py
COPY comet_gpt2.py comet_gpt2.py
COPY split/ split/

ENV PYTHONPATH=.
