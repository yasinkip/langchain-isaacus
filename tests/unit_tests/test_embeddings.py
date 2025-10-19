"""Test embedding model integration."""

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from langchain_isaacus import IsaacusEmbeddings

MODEL = "kanon-2-embedder"


def test_initialization_kanon_2_embedder() -> None:
    """Test embedding model initialization."""
    emb = IsaacusEmbeddings(api_key=SecretStr("NOT_A_VALID_KEY"), model=MODEL)  # type: ignore
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 128
    assert emb.model == MODEL


def test_initialization_with_dimension() -> None:
    emb = IsaacusEmbeddings(
        api_key=SecretStr("NOT_A_VALID_KEY"),  # type: ignore
        model=MODEL,
        dimensions=256,
        batch_size=10,
    )
    assert isinstance(emb, Embeddings)
    assert emb.model == MODEL
    assert emb.dimensions == 256
