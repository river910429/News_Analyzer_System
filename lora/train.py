"""QLoRA fine-tuning of Qwen2.5-7B-Instruct on the teacher's agent trajectories.

4-bit NF4 base weights (bitsandbytes), LoRA adapters on all attention and MLP projections,
gradient checkpointing; sized for a 16 GB GPU at 6k tokens (lower --max-len,
or use --base Qwen/Qwen2.5-3B-Instruct, if it runs out of memory).
Loss only on assistant turns (see sft_format.py).

    docker run --rm --gpus all -v ${PWD}/lora:/work/lora -v hf-cache:/root/.cache/huggingface `
        rag-trainer python lora/train.py   (PowerShell; see lora/README.md)
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sft_format import build_example, to_hf_messages
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent


def load(path: Path, tokenizer, max_len: int) -> Dataset:
    rows, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        ex = build_example(tokenizer, to_hf_messages(item["messages"]), item.get("tools"), max_len)
        if ex is None:
            dropped += 1
        else:
            rows.append(ex)
    lengths = sorted(len(r["input_ids"]) for r in rows)
    print(f"{path.name}: {len(rows)} examples, {dropped} longer than {max_len} tokens dropped; "
          f"median {lengths[len(lengths) // 2] if lengths else 0}, max {lengths[-1] if lengths else 0} tokens")
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--out", default=str(HERE / "out" / "adapter"))
    p.add_argument("--max-len", type=int, default=6144)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    data = Path(args.data_dir)
    train = load(data / "train.jsonl", tokenizer, args.max_len)
    val_path = data / "val.jsonl"
    val = load(val_path, tokenizer, args.max_len) if val_path.exists() and val_path.stat().st_size else None

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(args.out).parent / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch" if val else "no",
            save_strategy="epoch",
            save_total_limit=1,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (Path(args.out) / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"adapter -> {args.out}")


if __name__ == "__main__":
    main()
