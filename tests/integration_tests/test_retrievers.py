from typing import Type

from langchain_isaacus.retrievers import IsaacusRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestIsaacusRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[IsaacusRetriever]:
        """Get an empty vectorstore for unit tests."""
        return IsaacusRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
