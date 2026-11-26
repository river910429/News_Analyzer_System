import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lora"))

from sft_format import build_example, to_hf_messages  # noqa: E402


class ChatMLTokenizer:
    """Stand-in for the Qwen2.5 tokenizer: ChatML template, one token per character."""

    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=False):
        out = ""
        if tools:
            out += "<|im_start|>system\n<tools>" + json.dumps(tools) + "</tools><|im_end|>\n"
        for m in msgs:
            body = m.get("content", "")
            for c in m.get("tool_calls", []):
                body += "<tool_call>" + json.dumps(c["function"], ensure_ascii=False) + "</tool_call>"
            out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


OLLAMA_CONVERSATION = [
    {"role": "system", "content": "你是財報分析助理"},
    {"role": "user", "content": "瑞昱114年毛利率？"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_report", "arguments": {"query": "毛利率"}}}],
    },
    {"role": "tool", "tool_name": "search_report", "content": "毛利率48.2%"},
    {"role": "assistant", "content": "48.2%（來源：第 5 頁）"},
]


def test_to_hf_messages():
    hf = to_hf_messages(OLLAMA_CONVERSATION)
    assert hf[2]["tool_calls"][0] == {
        "type": "function",
        "function": {"name": "search_report", "arguments": {"query": "毛利率"}},
    }
    assert hf[3] == {"role": "tool", "name": "search_report", "content": "毛利率48.2%"}
    assert "tool_calls" not in hf[4]
    # arguments given as a JSON string are parsed
    raw = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}],
        }
    ]
    assert to_hf_messages(raw)[0]["tool_calls"][0]["function"]["arguments"] == {"expression": "1+1"}


def test_loss_only_on_assistant_turns():
    tok = ChatMLTokenizer()
    ex = build_example(tok, to_hf_messages(OLLAMA_CONVERSATION), tools=[{"name": "search_report"}], max_len=10_000)
    trained = "".join(chr(t) for t, label in zip(ex["input_ids"], ex["labels"], strict=True) if label != -100)
    assert '<tool_call>{"name": "search_report"' in trained
    assert "48.2%（來源：第 5 頁）<|im_end|>" in trained
    # the prompt, the question and the tool result are context only
    for context in ("你是財報分析助理", "瑞昱114年毛利率", "<tools>", "毛利率48.2%<|im_end|>"):
        assert context not in trained


def test_too_long_examples_are_dropped():
    assert build_example(ChatMLTokenizer(), to_hf_messages(OLLAMA_CONVERSATION), None, max_len=20) is None


def test_collect_keeps_only_clean_correct_trajectories():
    from collect import rejection_reason

    good = {
        "stopped_reason": "answered",
        "critiques": [],
        "corrections": [],
        "steps": [{"tool": "search_report"}],
        "answer": "毛利率為48.2%（來源：第 5 頁）",
    }
    item = {"answer": "48.2", "type": "lookup"}
    assert rejection_reason(item, good) == ""
    assert rejection_reason(item, good | {"answer": "毛利率為47.9%"}) == "wrong_answer"
    assert rejection_reason(item, good | {"critiques": ["…"], "checks": ["ungrounded"]}) == "critiqued:ungrounded"
    assert rejection_reason(item, good | {"corrections": [{"claimed": "1"}]}) == "arithmetic_fixed"
    assert rejection_reason(item, good | {"steps": [{"tool": "search_report", "recovered": True}]}) == "text_tool_call"
    assert rejection_reason(item, good | {"steps": [{"tool": "calculator"}]}) == "no_search"
    unanswerable = {"type": "unanswerable"}
    assert rejection_reason(unanswerable, good | {"answer": "根據目前的文件，無法得知此資訊"}) == ""
    assert rejection_reason(unanswerable, good) == "wrong_answer"
