"""Test Isaacus Reranks."""

from langchain_core.documents import Document

from langchain_isaacus import IsaacusRerank

# Please set ISAACUS_API_KEY in the environment variables
MODEL = "kanon-2-reranker"


def test_langchain_isaacus_rerank_documents() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [Document("foo bar")]
    rerank = IsaacusRerank(model=MODEL)
    output = rerank.compress_documents(documents, query)
    assert len(output) == 1


def test_langchain_isaacus_rerank_documents_multiple() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [Document("foo bar"), Document("bar foo"), Document("foo")]
    rerank = IsaacusRerank(model=MODEL)
    output = rerank.compress_documents(documents, query)
    assert len(output) == 3


async def test_langchain_isaacus_async_rerank_documents_multiple() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [Document("foo bar"), Document("bar foo"), Document("foo")]
    rerank = IsaacusRerank(model=MODEL)
    output = await rerank.acompress_documents(documents, query)
    assert len(output) == 3


def test_langchain_isaacus_rerank_documents_with_top_n() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [Document("foo bar"), Document("bar foo"), Document("foo")]
    rerank = IsaacusRerank(model=MODEL, top_n=2)
    output = rerank.compress_documents(documents, query)
    assert len(output) == 2