"""A small tool-calling agent loop (the "harness" around the LLM).

The model decides which tool to call; this loop executes it, feeds the result back,
and repeats until the model answers or the step budget runs out. Every step is
recorded so the run can be inspected, evaluated and debugged.
"""

import ast
import json
import logging
import operator
import re
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
    "比", "多多少", "少多少", "倍",
)
_REFUSAL = "無法得知"
# Ways a model says the document does not contain the answer.
_ABSTAIN = ("無法得知", "找不到", "並未", "未提及", "沒有提到", "未提供", "未明確")

# "a op b = c" written in the answer. No CJK inside, so it cannot span sentences.
_EXPR = re.compile(r"([0-9.,()+\-*/×÷%％ \t]+?)[ \t]*[=≈][ \t]*(-?[0-9][0-9,]*(?:\.[0-9]+)?)[ \t]*([%％])?")
# Any arithmetic between numbers, even when interrupted by labels: "95570 - 24917（技術人員）".
_DERIVATION = re.compile(r"[0-9][ \t]*[×÷*/][ \t]*[0-9]|[0-9][ \t]+[-+][ \t]+[0-9]")
_CN_QUARTER = {"一": "1", "二": "2", "三": "3", "四": "4"}


def _is_abstention(answer: str) -> bool:
    return any(a in answer for a in _ABSTAIN)


def check_arithmetic(answer: str) -> List[Dict[str, Any]]:
    """Recompute every "expression = result" in the answer. Returns the wrong ones.

    The claimed result counts as correct when it equals the true value rounded to the
    decimals the model wrote; "a / b = 34.39%" is also accepted for a ratio of 0.3439.
    """
    wrong = []
    for m in _EXPR.finditer(answer):
        expr = m.group(1).strip().lstrip("+-*/×÷ \t").replace("%", "").replace("％", "")
        if not re.search(r"[0-9].*[+\-*/×÷].*[0-9]", expr):
            continue
        result = calculator(expr)
        if result.startswith("錯誤"):
            continue
        actual = float(result)
        claimed_text = m.group(2)
        claimed = float(claimed_text.replace(",", ""))
        decimals = len(claimed_text.split(".")[1]) if "." in claimed_text else 0
        tolerance = 0.5 * 10**-decimals + 1e-9
        ok = abs(actual - claimed) <= tolerance or (m.group(3) and abs(actual * 100 - claimed) <= tolerance)
        if not ok:
            wrong.append({"expression": expr, "claimed": claimed_text, "actual": round(actual, 4)})
    return wrong


def _period_variants(question: str) -> List[List[str]]:
    """Periods the question asks about, each with the spellings a report may use.

    "115年第一季" -> [["115年第一季", "115年第1季", "2026年第一季", "2026年第1季", "2026Q1", "1Q26"]];
    "114年" -> [["114年", "2025年", "2025"]]. ROC year + 1911 = Gregorian year.
    """
    periods = []
    for m in re.finditer(r"(\d{2,4})\s*年(?:\s*第\s*([一二三四1-4])\s*季)?", question):
        year = int(m.group(1))
        if not (80 <= year <= 150 or 1990 <= year <= 2060):  # "過去10年" is a duration, not a year
            continue
        roc, ad = (year - 1911, year) if year > 1911 else (year, year + 1911)
        if m.group(2):
            q = _CN_QUARTER.get(m.group(2), m.group(2))
            cn = "一二三四"[int(q) - 1]
            variants = [f"{y}年第{x}季" for y in (roc, ad) for x in (q, cn)] + [f"{ad}Q{q}", f"{q}Q{str(ad)[2:]}"]
        else:
            variants = [f"{roc}年", f"{ad}年", str(ad)]
        asked = "".join(m.group(0).split())  # the question's own spelling goes first, for the feedback
        periods.append([asked] + [v for v in variants if v != asked])
    return periods


def critique(question: str, answer: str, steps: List[Dict[str, Any]], evidence: Optional[str] = None) -> Optional[str]:
    """Harness-side checks on a draft answer. Returns feedback for the model, or None to accept.

    Small local models skip the calculator and mis-calculate, give up after one search, or
    answer a question the document cannot answer with a number from another period or one
    they derived themselves. These checks catch each case and ask the model to try again.
    `evidence` is the full text of the search/page results (steps keep only a prefix).
    """
    used = [s["tool"] for s in steps]
    if evidence is None:
        evidence = "\n".join(str(s.get("result", "")) for s in steps if s["tool"] != "calculator")
    evidence = "".join(evidence.split())
    # A tool named in the answer but never called means the model only pretended to use it.
    faked = [name for name in ("calculator", "search_report", "get_page") if name in answer and name not in used]
    if faked:
        return (
            f"你在回答裡提到 {', '.join(faked)}，但並沒有真的呼叫這個工具。"
            "請發出工具呼叫拿到結果，最後的回答不要提到工具名稱。"
        )
    searches = used.count("search_report") + used.count("get_page")
    if _is_abstention(answer):
        if _REFUSAL in answer and searches < 2:
            return "你只查了一次就說查不到。請換個關鍵字再用 search_report 查一次，或用 get_page 讀相關頁面，再回答。"
        return None

    # The question asks about a period that never appears in what was retrieved.
    if searches:
        for variants in _period_variants(question):
            if not any(v in evidence for v in variants):
                return (
                    f"問題問的是「{variants[0]}」，但查到的資料裡沒有出現這個期間。"
                    "不要用其他期間的數字代替。請針對這個期間再查一次；文件確實沒有，"
                    "就回答「根據目前的文件，無法得知此資訊」。"
                )

    wrong = check_arithmetic(answer)
    if wrong:
        w = wrong[0]
        return (
            f"你的算式 {w['expression']} 算錯了：正確結果是 {w['actual']}，不是 {w['claimed']}。"
            "請用正確的結果重新回答，計算請呼叫 calculator。"
        )

    needs_calc = any(h in question for h in _CALC_HINTS)
    # A question that asks for a stated figure, answered with a figure the model worked out.
    if not needs_calc and _DERIVATION.search(answer):
        return (
            "這題問的是年報直接記載的數字，你卻用其他數字推算出來。年報沒有直接寫出這個數字時，"
            "正確的回答是「根據目前的文件，無法得知此資訊」；如果有直接寫，請照抄原文數字並註明頁碼。"
        )
    if needs_calc and "calculator" not in used and not _EXPR.search(answer):
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
    max_critiques: int = 3,
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
    evidence: List[str] = []  # full search/page results, for the critique
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
            feedback = None
            if len(critiques) < max_critiques:
                feedback = critique(question, answer, steps, "\n".join(evidence))
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
            if name != "calculator":
                evidence.append(result)
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
