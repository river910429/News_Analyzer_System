"""Annotate Chinese number formats with their Arabic value, for matching and for the LLM.

Taiwanese annual reports write years and large numbers in forms that neither BM25 nor a
7B model handles well:

    民國一一四年 / 民國一百一十四年   -> 114年   (query terms say "114年")
    9萬557人                          -> 90,557   (qwen2.5 read it as 95570)
    1,224億2,000萬美元                 -> 1,224.2億 = 122,420,000,000
    3兆8,090億5,000萬                  -> 38,090.5億

The original text is kept; the value is appended in 〔〕 so citations and keyword-based
evaluation still see the source wording. Running it twice changes nothing.
This file is shared by the nlp-worker (index time) and the api-gateway (tool output).
"""

import re

_DIGITS = {
    **{"零": 0, "〇": 0, "○": 0, "一": 1, "二": 2, "兩": 2, "三": 3},
    **{"四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9},
}
_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUM = "零〇○一二三四五六七八九十百千"

_YEAR = re.compile(rf"([{_CN_NUM}]{{2,6}})年(?!〔)")
_NUM = r"\d[\d,]*(?:\.\d+)?"
# 3兆8,090億5,000萬1,234 / 1,224億2,000萬 / 9萬557 — at least two parts, or 兆/億 with a remainder.
_MIXED = re.compile(rf"(?<![\d.,])(?:({_NUM})兆)?(?:({_NUM})億)?(?:({_NUM})萬)?(\d[\d,]*)?(?![\d,.]*[萬億兆〔])")


def cn_to_int(text: str) -> int:
    """一一四 -> 114 (digit by digit), 一百一十四 -> 114, 十四 -> 14, 一百零五 -> 105."""
    if all(c in _DIGITS for c in text):
        return int("".join(str(_DIGITS[c]) for c in text))
    total, current = 0, 0
    for c in text:
        if c in _DIGITS:
            current = _DIGITS[c]
        elif c in _UNITS:
            total += (current or 1) * _UNITS[c]
            current = 0
        else:
            raise ValueError(text)
    return total + current


def _year(m: re.Match) -> str:
    try:
        value = cn_to_int(m.group(1))
    except ValueError:
        return m.group(0)
    if 80 <= value <= 150:  # ROC years; skips 十年, 三年 and the like
        return f"{m.group(0)}〔{value}年〕"
    return m.group(0)


def _num(s: str) -> float:
    return float(s.replace(",", "")) if s else 0.0


def _fmt(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _mixed(m: re.Match) -> str:
    trillion, hundred_m, ten_k, rest = m.groups()
    parts = [p for p in (trillion, hundred_m, ten_k) if p]
    if not parts:
        return m.group(0)
    # A lone "10億" or "387萬" is clear enough; annotate only combined forms.
    if len(parts) + bool(rest) < 2:
        return m.group(0)
    value = _num(trillion) * 1e12 + _num(hundred_m) * 1e8 + _num(ten_k) * 1e4 + _num(rest)
    if value >= 1e8:
        return f"{m.group(0)}〔{_fmt(value / 1e8)}億 = {value:,.0f}〕"
    return f"{m.group(0)}〔{value:,.0f}〕"


def annotate_numbers(text: str) -> str:
    if not text:
        return text
    text = _YEAR.sub(_year, text)
    return _MIXED.sub(_mixed, text)


_ANNOTATION = re.compile(r"〔[^〕]*〕")


def strip_annotations(text: str) -> str:
    return _ANNOTATION.sub("", text)
