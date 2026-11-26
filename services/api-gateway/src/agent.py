"""A small tool-calling agent loop (the "harness" around the LLM).

The model decides which tool to call; this loop executes it, feeds the result back,
and repeats until the model answers or the step budget runs out. Every step is
recorded so the run can be inspected, evaluated and debugged.
"""

import ast
import json
import logging
import operator
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("api-gateway.agent")

SYSTEM_PROMPT = """你是財報分析助理，透過工具查詢使用者上傳的財報來回答問題。

規則：
1. 先用 search_report 查資料，不要憑記憶回答。比較兩家以上公司時，每家公司分別呼叫一次
   search_report，company 參數填公司名稱（例如「台積電」）。
2. 問題牽涉兩個數字時（今年與去年、總數與部分），確認兩個數字都查到；沒查到就換關鍵字再查，
   或用 get_page 讀整頁。
3. 只要能從查到的數字算出答案就要計算，所有加減乘除、比例、百分點差異都呼叫 calculator，
   算式裡的數字不要有逗號。
4. 財報表格的年度欄位順序因公司而異，先看表頭的年度標示再取數字；同一頁可能同時有「合併」與
   「個體」報表，計算用的數字必須來自同一張表；只用問題所問公司的資料（看檔名）。
5. 工具呼叫一定要真的發出，不要把呼叫寫在回答文字裡。
6. 回答列出用到的數字與算式，數字照抄並註明單位，最後用「（來源：第 N 頁）」標註頁碼；
   回答中不要提到工具名稱，也不要描述呼叫過程。
7. 確實查了兩次以上仍找不到，才回答「根據目前的文件，無法得知此資訊」。

請用繁體中文簡潔回答。"""

# Words that signal the question needs arithmetic.
_CALC_HINTS = (
    "增加", "減少", "相差", "差距", "提高", "下降", "占", "比例", "百分之", "百分點", "合計", "減去", "多少倍",
)
_REFUSAL = "無法得知"


def critique(question: str, answer: str, steps: List[Dict[str, Any]]) -> Optional[str]:
    """Harness-side checks on a draft answer. Returns feedback for the model, or None to accept.

    Small local models often skip the calculator (and then mis-calculate) or give up after one
    search. These checks catch both and ask the model to try again.
    """
    used = [s["tool"] for s in steps]
    # A tool named in the answer but never called means the model only pretended to use it.
    faked = [name for name in ("calculator", "search_report", "get_page") if name in answer and name not in used]
    if faked:
        return (
            f"你在回答裡提到 {', '.join(faked)}，但並沒有真的呼叫這個工具。"
            "請發出工具呼叫拿到結果，最後的回答不要提到工具名稱。"
        )
    if _REFUSAL in answer and used.count("search_report") + used.count("get_page") < 2:
        return "你只查了一次就說查不到。請換個關鍵字再用 search_report 查一次，或用 get_page 讀相關頁面，再回答。"
    if any(h in question for h in _CALC_HINTS) and "calculator" not in used and _REFUSAL not in answer:
        return "這題需要計算，但你還沒有使用 calculator。請把查到的數字用 calculator 算出結果，再根據計算結果回答。"
    return None


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_report",
            "description": "在已上傳的財報中搜尋與問題相關的段落（混合檢索）。回傳段落內容與頁碼。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜尋的內容，例如「114年毛利率」"},
                    "top_k": {"type": "integer", "description": "回傳幾段，預設 5"},
                    "company": {
                        "type": "string",
                        "description": "只搜尋這家公司的年報，例如「台積電」、「聯發科」、「聯電」",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page",
            "description": "讀取某份文件某一頁的完整文字，用於 search_report 的片段不夠完整時。",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer", "description": "文件 id，search_report 的結果會附上"},
                    "page": {"type": "integer", "description": "頁碼"},
                },
                "required": ["document_id", "page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "計算數學算式，支援 + - * / ** 和括號，例如「59.9 - 56.1」或「2472228595 / 7933023878」",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "算式，數字不要有逗號"}},
                "required": ["expression"],
            },
        },
    },
]

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> str:
    """Evaluate arithmetic safely (no eval): only numbers and + - * / ** % ( )."""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    cleaned = expression.replace(",", "").replace("％", "%").replace("×", "*").replace("÷", "/")
    try:
        value = _eval(ast.parse(cleaned, mode="eval"))
    except ZeroDivisionError:
        return "錯誤：除以零"
    except (SyntaxError, ValueError) as e:
        return f"錯誤：無法計算 {expression!r}（{e}）"
    if isinstance(value, float):
        value = round(value, 6)
        if value.is_integer():
            value = int(value)
    return str(value)


# chat_fn(messages, tools) -> assistant message: {"content": str, "tool_calls": [...]} (Ollama format)
ChatFn = Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]]


def _scalar(value: Any) -> Any:
    """Unwrap values a small model nests by mistake.

    qwen2.5 sometimes copies the JSON schema into the call, sending
    {"company": {"type": "string", "value": "聯電"}} or {"query": {"text": "..."}}
    instead of plain strings.
    """
    while isinstance(value, dict):
        for key in ("value", "text", "query", "company", "expression", "content"):
            if key in value:
                value = value[key]
                break
        else:
            strings = [v for v in value.values() if isinstance(v, (str, int, float))]
            value = strings[0] if strings else ""
    return value


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if not isinstance(raw, dict):
        return {}
    args = {k: _scalar(v) for k, v in raw.items()}
    if isinstance(args.get("top_k"), str) and args["top_k"].isdigit():
        args["top_k"] = int(args["top_k"])
    return args


def run_agent(
    question: str,
    chat_fn: ChatFn,
    tools: Dict[str, Callable[..., str]],
    max_steps: int = 6,
    max_critiques: int = 2,
) -> Dict[str, Any]:
    """Run the tool-calling loop. Returns {"answer", "steps", "critiques", "stopped_reason", "elapsed_ms"}.

    max_steps limits LLM calls; max_critiques limits how often a draft answer is sent back.
    """
    started = time.perf_counter()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: List[Dict[str, Any]] = []
    critiques: List[str] = []
    stopped_reason = "answered"
    answer = ""

    for _ in range(max_steps):
        try:
            reply = chat_fn(messages, TOOL_SCHEMAS)
        except Exception as e:  # LLM timeout or connection error: stop, keep the trace
            logger.exception("LLM call failed")
            stopped_reason = "llm_error"
            answer = f"模型呼叫失敗，無法完成回答（{e.__class__.__name__}）。"
            break
        tool_calls = reply.get("tool_calls") or []
        messages.append({"role": "assistant", "content": reply.get("content", ""), "tool_calls": tool_calls})

        if not tool_calls:
            answer = (reply.get("content") or "").strip()
            feedback = critique(question, answer, steps) if len(critiques) < max_critiques else None
            if feedback is None:
                break
            logger.info("agent critique: %s", feedback)
            critiques.append(feedback)
            messages.append({"role": "user", "content": feedback})
            continue

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = _parse_arguments(fn.get("arguments"))
                if name not in tools:
                    raise KeyError(f"unknown tool {name!r}, available: {', '.join(tools)}")
                result = tools[name](**args)
            except Exception as e:  # report the error to the model so it can correct itself
                args = fn.get("arguments")
                result = f"工具錯誤：{e.__class__.__name__}: {e}"
            logger.info("agent tool %s(%s) -> %d chars", name, args, len(result))
            steps.append({"tool": name, "arguments": args, "result": result[:2000]})
            messages.append({"role": "tool", "tool_name": name, "content": result})
    else:
        # Step budget used up: ask for a final answer from what was gathered, without tools.
        stopped_reason = "max_steps"
        messages.append({"role": "user", "content": "請根據目前查到的資料直接回答，不要再呼叫工具。"})
        answer = (chat_fn(messages, []).get("content") or "").strip()

    return {
        "answer": answer or "無法生成回答",
        "steps": steps,
        "critiques": critiques,
        "stopped_reason": stopped_reason,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
