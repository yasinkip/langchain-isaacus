"""Test Isaacus embeddings."""

from typing import Type

from langchain_isaacus.embeddings import IsaacusEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[IsaacusEmbeddings]:
        return IsaacusEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
