from src.agent import calculator, run_agent


def test_calculator_basic_and_commas():
    assert calculator("59.9 - 56.1") == "3.8"
    assert calculator("3,809,054,272 - 2,894,307,699") == "914746573"
    assert calculator("(78 - 19) * 2") == "118"
    assert calculator("2472228595 / 7933023878 * 100").startswith("31.16")


def test_calculator_rejects_code_and_bad_input():
    assert calculator("__import__('os').system('ls')").startswith("錯誤")
    assert calculator("1 / 0") == "錯誤：除以零"
    assert calculator("2 ** 1000").startswith("錯誤")


def scripted_chat(replies):
    """A fake LLM that returns pre-written replies and records what it was sent."""
    calls = []

    def chat(messages, tools):
        calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return replies[len(calls) - 1]

    return chat, calls


def tool_call(name, **arguments):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def test_agent_searches_calculates_then_answers():
    chat, calls = scripted_chat(
        [
            tool_call("search_report", query="毛利率"),
            tool_call("calculator", expression="59.9 - 56.1"),
            {"content": "毛利率增加 3.8 個百分點（來源：第 7 頁）"},
        ]
    )
    tools = {
        "search_report": lambda query, top_k=5: "[1] 文件 1 第 7 頁\n毛利率為59.9%，前一年為56.1%",
        "calculator": calculator,
    }
    result = run_agent("毛利率增加幾個百分點？", chat, tools)

    assert result["answer"].startswith("毛利率增加 3.8")
    assert [s["tool"] for s in result["steps"]] == ["search_report", "calculator"]
    assert result["steps"][1]["result"] == "3.8"
    assert result["stopped_reason"] == "answered"
    # the calculator result was fed back to the model
    last_messages = calls[2]["messages"]
    assert last_messages[-1] == {"role": "tool", "tool_name": "calculator", "content": "3.8"}


def test_agent_reports_tool_errors_back_to_the_model():
    chat, calls = scripted_chat([tool_call("unknown_tool", x=1), {"content": "抱歉"}])
    result = run_agent("?", chat, {"calculator": calculator})
    assert "工具錯誤" in result["steps"][0]["result"]
    assert "unknown_tool" in calls[1]["messages"][-1]["content"]


def test_agent_accepts_json_string_arguments():
    chat, _ = scripted_chat(
        [
            {"content": "", "tool_calls": [{"function": {"name": "calculator", "arguments": '{"expression": "1+2"}'}}]},
            {"content": "3"},
        ]
    )
    result = run_agent("1+2?", chat, {"calculator": calculator})
    assert result["steps"][0]["result"] == "3"


def test_agent_stops_at_max_steps_and_answers_without_tools():
    replies = [tool_call("calculator", expression="1+1")] * 3 + [{"content": "最終答案"}]
    chat, calls = scripted_chat(replies)
    result = run_agent("?", chat, {"calculator": calculator}, max_steps=3)
    assert result["stopped_reason"] == "max_steps"
    assert result["answer"] == "最終答案"
    assert calls[-1]["tools"] == []  # final call has no tools


def test_critique_sends_back_answer_that_skipped_the_calculator():
    chat, calls = scripted_chat(
        [
            tool_call("search_report", query="EPS"),
            {"content": "增加了 20 元"},  # mental arithmetic, no calculator
            tool_call("calculator", expression="66.25 - 45.25"),
            {"content": "增加 21 元（來源：第 7 頁）"},
        ]
    )
    tools = {"search_report": lambda query, top_k=5: "EPS 66.25 元，前一年 45.25 元", "calculator": calculator}
    result = run_agent("每股盈餘比前一年增加了多少元？", chat, tools)

    assert result["answer"].startswith("增加 21 元")
    assert len(result["critiques"]) == 1 and "calculator" in result["critiques"][0]
    assert calls[2]["messages"][-1]["role"] == "user"  # feedback was sent to the model


def test_critique_asks_to_search_again_before_giving_up():
    chat, _ = scripted_chat(
        [
            tool_call("search_report", query="2奈米"),
            {"content": "根據目前的文件，無法得知此資訊"},
            tool_call("search_report", query="A14 量產"),
            {"content": "相差 3 年"},
        ]
    )
    tools = {"search_report": lambda query, top_k=5: "片段", "calculator": calculator}
    result = run_agent("2奈米與A14量產相隔幾年？", chat, tools)
    assert [s["tool"] for s in result["steps"]] == ["search_report", "search_report"]
    assert "換個關鍵字" in result["critiques"][0]


def test_critiques_are_limited():
    chat, _ = scripted_chat([{"content": "無法得知"}] * 3)
    result = run_agent("差多少？", chat, {"calculator": calculator}, max_critiques=2)
    assert len(result["critiques"]) == 2
    assert result["answer"] == "無法得知"


def test_critique_catches_tool_calls_written_as_text():
    from src.agent import critique

    feedback = critique("股利提高多少？", 'calculator("6.0 - 5.0") 得到 1.0，提高 1.0 元', [])
    assert feedback is not None and "工具呼叫寫在回答文字裡" in feedback
