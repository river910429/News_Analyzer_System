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
