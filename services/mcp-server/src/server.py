"""MCP server: exposes the financial-report RAG system as tools for MCP clients.

Any MCP client (Claude Code, Claude Desktop, MCP Inspector, other agents) can list
documents, search them and read pages. The server only calls the api-gateway's HTTP
API, so retrieval logic lives in one place.

Run inside docker-compose (streamable HTTP on :8001/mcp), or locally:
    API_URL=http://localhost:8000 python src/server.py
"""

import os

import httpx
from mcp.server.mcpserver import MCPServer

API_URL = os.getenv("API_URL", "http://api-gateway:8000")
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8001"))
TIMEOUT = float(os.getenv("MCP_API_TIMEOUT", "60"))

mcp = MCPServer(
    name="financial-report-rag",
    instructions=(
        "Search uploaded financial reports (mostly Traditional Chinese annual reports). "
        "Use search_report first, then get_page for the full page when a passage is cut off. "
        "Always cite the page number."
    ),
)


def _api(method: str, path: str, **kwargs):
    response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def list_documents() -> str:
    """List uploaded reports with their document id and indexing status."""
    docs = _api("GET", "/documents")
    if not docs:
        return "No documents uploaded yet."
    return "\n".join(f"- id={d['id']} {d.get('name') or d.get('filename')} ({d['status']})" for d in docs)


@mcp.tool()
def search_report(query: str, top_k: int = 5, company_filter: bool = True) -> str:
    """Search the reports with hybrid retrieval (vector + BM25). Returns passages with document id and page.

    Args:
        query: what to look for, e.g. "台積電 114年毛利率"; naming a company restricts the search to its report
        top_k: number of passages to return (1-10)
        company_filter: set False to search all reports even if the query names a company
    """
    payload = {"query": query, "top_k": max(1, min(top_k, 10)), "company_filter": company_filter}
    results = _api("POST", "/retrieve", json=payload)
    if not results:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{i}] document {r.get('document_id')} ({r['filename']}), page {r.get('page')}, "
        f"similarity {r['similarity_score']:.2f}\n{r['content']}"
        for i, r in enumerate(results, start=1)
    )


@mcp.tool()
def get_page(document_id: int, page: int) -> str:
    """Read the full text of one page of a report.

    Args:
        document_id: id from list_documents or search_report
        page: page number from search_report
    """
    try:
        return _api("GET", f"/documents/{document_id}/pages/{page}")["text"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Document {document_id} has no indexed text on page {page}."
        raise


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
