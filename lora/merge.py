"""Merge the LoRA adapter into the base weights and save a standalone fp16 model.

Runs on the CPU (needs ~16 GB of RAM for 7B), so the GPU stays free. The merged model is
then imported into Ollama by export.ps1.

    docker run --rm -v ${PWD}/lora:/work/lora -v hf-cache:/root/.cache/huggingface rag-trainer python lora/merge.py
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=str(HERE / "out" / "adapter"))
    p.add_argument("--out", default=str(HERE / "out" / "merged"))
    args = p.parse_args()

    # No device_map: with one, PEFT re-dispatches the model through accelerate when loading the
    # adapter, and accelerate's memory estimate asked for disk offload of half the layers.
    # A plain CPU load needs ~15 GB of RAM for 7B in fp16.
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter, torch_device="cpu").merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
