"""Test embedding model integration."""

from typing import Type

from langchain_isaacus.embeddings import IsaacusEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[IsaacusEmbeddings]:
        return IsaacusEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
