"""Test embedding model integration."""

from langchain_core.documents import BaseDocumentCompressor
from pydantic import SecretStr

from langchain_isaacus import IsaacusRerank

MODEL = "kanon-2-reranker"


def test_initialization_kanon_2_reranker() -> None:
    """Test embedding model initialization."""
    rr = IsaacusRerank(api_key=SecretStr("NOT_A_VALID_KEY"), model=MODEL)
    assert isinstance(rr, BaseDocumentCompressor)
    assert rr.top_n is None
    assert rr.model == MODEL


def test_initialization_with_top_n() -> None:
    top_n = 5
    rr = IsaacusRerank(
        api_key=SecretStr("NOT_A_VALID_KEY"),
        model=MODEL,
        top_n=top_n,
    )
    assert isinstance(rr, IsaacusRerank)
    assert rr.top_n == top_n
    assert rr.model == MODEL
