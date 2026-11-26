"""The API module must import (catches broken imports before a deploy)."""


def test_api_module_imports_and_routes_exist():
    from src import main

    paths = {route.path for route in main.app.routes}
    assert {"/health", "/upload", "/documents", "/search", "/retrieve"} <= paths


def test_build_prompt_numbers_sources_and_shows_pages():
    from src.main import SearchResult, build_prompt

    sources = [
        SearchResult(filename="a.pdf", page=7, content="毛利率為59.9%", similarity_score=0.8),
        SearchResult(filename="a.pdf", page=None, content="EPS 66.25", similarity_score=0.7),
    ]
    prompt = build_prompt("毛利率？", sources)
    assert "[參考資料 1]（a.pdf，第 7 頁）" in prompt
    assert "[參考資料 2]（a.pdf）" in prompt
    assert "毛利率？" in prompt


def test_search_stream_sends_sources_tokens_and_done(monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from src import main

    source = main.SearchResult(filename="a.pdf", page=7, content="毛利率為59.9%", similarity_score=0.8)
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [source])
    monkeypatch.setattr(main, "generate_stream", lambda prompt: iter(["毛利率", "為59.9%"]))

    res = TestClient(main.app).post("/search/stream", json={"query": "毛利率？"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[0]["sources"][0]["page"] == 7
    assert "".join(e["text"] for e in events if e["type"] == "token") == "毛利率為59.9%"


def test_search_stream_without_sources_answers_politely(monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from src import main

    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    res = TestClient(main.app).post("/search/stream", json={"query": "天氣如何？"})
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[1] == {"type": "token", "text": main.NO_RESULT_ANSWER}
    assert events[-1]["type"] == "done"


def test_generate_stream_parses_ollama_ndjson(monkeypatch):
    from src import main

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield b'{"response": "\\u6bdb\\u5229", "done": false}'
            yield b""
            yield b'{"response": "\\u7387", "done": false}'
            yield b'{"response": "", "done": true}'
            yield b'{"response": "ignored", "done": false}'

    monkeypatch.setattr(main.requests, "post", lambda *a, **k: FakeResponse())
    assert list(main.generate_stream("prompt")) == ["毛利", "率"]


def test_merge_chunks_removes_overlap_lines():
    from src.main import merge_chunks

    chunks = ["第一行\n第二行\n第三行", "第二行\n第三行\n第四行", "第五行"]
    assert merge_chunks(chunks) == "第一行\n第二行\n第三行\n第四行\n第五行"


def test_agent_endpoint_returns_answer_and_steps(monkeypatch):
    from fastapi.testclient import TestClient

    from src import main

    replies = iter(
        [
            {"content": "", "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "2*3"}}}]},
            {"content": "答案是 6"},
        ]
    )
    monkeypatch.setattr(main, "ollama_chat", lambda messages, tools, model=None: next(replies))

    res = TestClient(main.app).post("/agent", json={"question": "2 乘 3？"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "答案是 6"
    assert body["steps"][0] == {
        "tool": "calculator",
        "arguments": {"expression": "2*3"},
        "result": "6",
        "recovered": False,
    }


def test_delete_route_exists():
    from src import main

    routes = {(route.path, tuple(sorted(route.methods))) for route in main.app.routes if hasattr(route, "methods")}
    assert ("/documents/{doc_id}", ("DELETE",)) in routes
