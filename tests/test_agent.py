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
    assert feedback is not None and "並沒有真的呼叫" in feedback
    # narrated in Chinese, as qwen2.5 did after the call syntax was removed from the prompt
    narrated = "計算過程：呼叫 calculator 工具，算式為 59.9 - 56.1，得到 3.8"
    steps = [{"tool": "search_report"}]
    assert "並沒有真的呼叫" in critique("毛利率增加幾個百分點？", narrated, steps)
    # mentioning a tool that really was called is fine
    assert critique("毛利率增加幾個百分點？", narrated, steps + [{"tool": "calculator"}]) is None


def test_llm_failure_stops_with_trace_instead_of_raising():
    calls = {"n": 0}

    def chat(messages, tools):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("read timed out")
        return tool_call("calculator", expression="1+1")

    result = run_agent("?", chat, {"calculator": calculator})
    assert result["stopped_reason"] == "llm_error"
    assert result["steps"][0]["result"] == "2"
    assert "TimeoutError" in result["answer"]


def test_nested_schema_like_arguments_are_unwrapped():
    from src.agent import _parse_arguments

    raw = {"company": {"type": "string", "value": "聯電"}, "query": {"text": "負債總額"}, "top_k": "5"}
    assert _parse_arguments(raw) == {"company": "聯電", "query": "負債總額", "top_k": 5}
    assert _parse_arguments('{"company": {"company": "聯發科"}}') == {"company": "聯發科"}


def test_agent_tool_receives_plain_strings_from_nested_call():
    chat, _ = scripted_chat(
        [
            tool_call("search_report", company={"type": "string", "value": "聯電"}, query={"text": "毛利率"}),
            {"content": "聯電毛利率 29.0%（來源：第 4 頁）"},
        ]
    )
    seen = {}

    def search_report(query, top_k=5, company=""):
        seen.update(query=query, company=company)
        return "聯電毛利率29.0%"

    run_agent("聯電毛利率？", chat, {"search_report": search_report, "calculator": calculator})
    assert seen == {"query": "毛利率", "company": "聯電"}


# --- checks added after the multi-company run (eval/results/*_agent-fix.json) ---


def test_arithmetic_check_catches_wrong_results_from_the_run():
    from src.agent import check_arithmetic

    # x-08: the model wrote 34.57; the true value is 34.39
    wrong = check_arithmetic("199,140,569 / 578,996,009 * 100 = 34.57%")
    assert wrong and wrong[0]["actual"] == 34.3941
    # x-02: "* 100% ≈ 29.1%" is really 29.0068
    assert check_arithmetic("毛利率為68,906,499 / 237,553,199 * 100% ≈ 29.1%（來源：第 50 頁）")
    # correct ones pass, including a ratio written as a percentage and a list bullet
    assert check_arithmetic("66.25 - 29.00 = 37.25（來源：第 7 頁）") == []
    assert check_arithmetic("199140569 / 578996009 = 34.39%") == []
    assert check_arithmetic("- 59.9 - 29.0 = 30.9") == []
    assert check_arithmetic("第 7 頁，2025-12-31") == []


def test_critique_gives_the_correct_value_for_wrong_arithmetic():
    from src.agent import critique

    steps = [{"tool": "search_report", "result": "114年 負債總計 199,140,569 資產總計 578,996,009"}]
    feedback = critique("聯電114年負債總額約占資產總額的百分之幾？", "199140569 / 578996009 * 100 = 34.57%", steps)
    assert "34.3941" in feedback


def test_critique_rejects_a_period_that_was_never_retrieved():
    from src.agent import critique

    # u-02: asked about 115年第一季, answered with the full-year 114 figure
    steps = [{"tool": "search_report", "result": "114年 營業收入 237,553,199"}] * 2
    feedback = critique("聯電115年第一季的營收是多少？", "聯電 115 年第一季的營收為 237,553,199 千元", steps)
    assert feedback and "115年第一季" in feedback
    # the Gregorian spelling of the same year counts
    steps = [{"tool": "search_report", "result": "2025年 毛利率 59.9%"}]
    assert critique("台積電114年的毛利率是多少？", "59.9%（來源：第 7 頁）", steps) is None
    # durations are not periods
    steps = [{"tool": "search_report", "result": "研發投入"}]
    assert critique("過去10年研發投入的重點是什麼？", "先進製程（來源：第 3 頁）", steps) is None


def test_critique_rejects_derived_numbers_for_a_lookup_question():
    from src.agent import critique

    steps = [{"tool": "search_report", "result": "2025年 員工 95570 人"}] * 2
    # u-03: subtracted staff categories to "find" a plant's headcount
    answer = "95570 - 24917（技術人員） - 11368（助理人員） = 59285"
    assert "推算" in critique("台積電熊本晶圓廠114年的員工人數是多少？", answer, steps)
    # abstaining is the right answer
    assert critique("台積電熊本晶圓廠114年的員工人數是多少？", "財報中並未提到熊本廠的員工人數", steps) is None


def test_critique_runs_on_full_results_not_the_truncated_steps():
    chat, _ = scripted_chat(
        [
            tool_call("search_report", query="營收"),
            {"content": "營收 100 億（來源：第 3 頁）"},
        ]
    )
    # "115年" sits past the 2000-character prefix kept in steps
    tools = {"search_report": lambda query, top_k=5, company="": "x" * 2500 + "115年第一季營收 100 億"}
    result = run_agent("115年第一季的營收是多少？", chat, tools)
    assert result["critiques"] == []
