# Training image for the LoRA stage. Needs Docker Desktop with the WSL2 backend and an NVIDIA
# driver; run with --gpus all. Base weights are cached in the hf-cache volume.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONPATH=/work/lora HF_HUB_ENABLE_HF_TRANSFER=0
