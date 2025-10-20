"""Test Isaacus embeddings."""

from langchain_isaacus import IsaacusEmbeddings

# Please set ISAACUS_API_KEY in the environment variables
MODEL = "kanon-2-embedder"


def test_langchain_isaacus_embedding_documents() -> None:
    """Test Isaacus embeddings."""
    documents = ["foo bar"]
    embedding = IsaacusEmbeddings(model=MODEL)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1792


def test_langchain_isaacus_embedding_documents_multiple() -> None:
    """Test Isaacus embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = IsaacusEmbeddings(model=MODEL, batch_size=2)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1792
    assert len(output[1]) == 1792
    assert len(output[2]) == 1792


def test_langchain_isaacus_embedding_query() -> None:
    """Test Isaacus embeddings."""
    document = "foo bar"
    embedding = IsaacusEmbeddings(model=MODEL)
    output = embedding.embed_query(document)
    assert len(output) == 1792


async def test_langchain_isaacus_async_embedding_documents_multiple() -> None:
    """Test Isaacus embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = IsaacusEmbeddings(model=MODEL, batch_size=2)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1792
    assert len(output[1]) == 1792
    assert len(output[2]) == 1792


async def test_langchain_isaacus_async_embedding_query() -> None:
    """Test Isaacus embeddings."""
    document = "foo bar"
    embedding = IsaacusEmbeddings(model=MODEL)
    output = await embedding.aembed_query(document)
    assert len(output) == 1792


def test_langchain_isaacus_embedding_documents_with_output_dimension() -> None:
    """Test Isaacus embeddings."""
    documents = ["foo bar"]
    embedding = IsaacusEmbeddings(model=MODEL, dimensions=256)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 256
