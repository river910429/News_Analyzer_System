"""Make service code importable in tests without Docker or ML models."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "nlp-worker" / "src"))  # chunking
sys.path.insert(0, str(ROOT / "services" / "api-gateway"))  # src.main, src.lexical
sys.path.insert(0, str(ROOT / "eval"))  # run_eval


class FakeSentenceTransformer:
    """Stands in for the 2 GB embedding model: deterministic 1024-dim vectors."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        import numpy as np

        if isinstance(texts, str):
            return np.zeros(1024)
        return np.zeros((len(texts), 1024))


@pytest.fixture(autouse=True, scope="session")
def fake_sentence_transformers():
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    module.CrossEncoder = FakeSentenceTransformer
    sys.modules.setdefault("sentence_transformers", module)
    yield
