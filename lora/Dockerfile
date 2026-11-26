# Training image for the LoRA stage. Needs Docker Desktop with the WSL2 backend and an NVIDIA
# driver; run with --gpus all. Base weights are cached in the hf-cache volume.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
# Keep the image's torch (pip would otherwise upgrade it for a newer dependency), and remove
# torchvision: it is built against that exact torch, transformers imports it when present,
# and a mismatch fails with "operator torchvision::nms does not exist".
RUN pip install --no-cache-dir -r /tmp/requirements.txt "torch==2.5.1" \
    && pip uninstall -y torchvision torchaudio
ENV PYTHONPATH=/work/lora HF_HUB_ENABLE_HF_TRANSFER=0
