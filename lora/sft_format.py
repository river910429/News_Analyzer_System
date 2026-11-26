"""Turn agent conversations into supervised fine-tuning examples.

The api-gateway records conversations in Ollama's message format. Training uses the
Hugging Face chat template of the base model (Qwen2.5), so messages are converted first.
Loss is computed only on the assistant turns — the tool calls and the final answer — never on
the system prompt, the question or the tool results, which the model only has to read.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def to_hf_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ollama format -> Hugging Face chat-template format.

    Ollama: {"role": "tool", "tool_name": ..., "content": ...} and assistant tool_calls of
    {"function": {"name", "arguments"}} with arguments as a dict (or a JSON string).
    """
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            msg: Dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                calls.append({"type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "name": m.get("tool_name", ""), "content": m.get("content", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def assistant_spans(
    tokenizer, messages: List[Dict[str, Any]], tools: Optional[list]
) -> Tuple[str, List[Tuple[int, int]]]:
    """Render the conversation and return (text, [(start, end) char spans of assistant turns]).

    For assistant message i, the span runs from the end of the rendered prompt for
    messages[:i] (with the generation prompt "<|im_start|>assistant\\n") to the end of the
    rendered messages[:i+1], so it covers the model's output including <|im_end|>.
    """

    def render(msgs, generation_prompt=False):
        return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=generation_prompt)

    text = render(messages)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prompt = render(messages[:i], generation_prompt=True)
        upto = render(messages[: i + 1])
        if not (text.startswith(upto) and upto.startswith(prompt)):
            raise ValueError("chat template is not prefix-stable; cannot locate assistant turns")
        spans.append((len(prompt), len(upto)))
    return text, spans


def build_example(tokenizer, messages: List[Dict[str, Any]], tools: Optional[list], max_len: int) -> Optional[dict]:
    """input_ids / attention_mask / labels with -100 outside assistant turns. None if too long.

    Long examples are dropped rather than truncated: a cut-off conversation would teach the
    model to stop mid tool call.
    """
    text, spans = assistant_spans(tokenizer, messages, tools)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = []
    for tok_id, (start, end) in zip(ids, enc["offset_mapping"], strict=True):
        inside = any(s <= start and end <= e for s, e in spans) and end > start
        labels.append(tok_id if inside else -100)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
