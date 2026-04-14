from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Sequence, Union

from isaacus import AsyncIsaacus, Isaacus
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import PrivateAttr, SecretStr


class IsaacusRerank(BaseDocumentCompressor):
    """Document compressor that uses the `Isaacus Rerank API`."""

    model: str
    """Model to use for reranking."""

    top_n: Optional[int] = None
    """ Number of documents to return. Defaults to returning all documents."""

    api_key: Optional[SecretStr] = None
    """API key for Isaacus.
    Can also be set via the `ISAACUS_API_KEY` environment variable."""

    base_url: Optional[str] = None
    """Base URL for the Isaacus API."""

    _client: Optional[Isaacus] = PrivateAttr(default=None)
    """Isaacus client instance. Instantiated lazily."""

    _aclient: Optional[AsyncIsaacus] = PrivateAttr(default=None)
    """Isaacus async client instance. Instantiated lazily."""

    def _document_to_str(self, doc: Union[Document, str]) -> str:
        if isinstance(doc, Document):
            return doc.page_content
        return doc

    def _get_api_key(self) -> Optional[str]:
        return self.api_key.get_secret_value() if self.api_key else None

    def _get_client(self) -> Isaacus:
        if not self._client:
            self._client = Isaacus(
                api_key=self._get_api_key(),
                base_url=self.base_url,
            )

        return self._client

    def _get_aclient(self) -> AsyncIsaacus:
        if not self._aclient:
            self._aclient = AsyncIsaacus(
                api_key=self._get_api_key(),
                base_url=self.base_url,
            )

        return self._aclient

    def rerank(
        self, documents: Sequence[Union[Document, str]], query: str
    ) -> list[dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to
        the query.
        """
        docs = [self._document_to_str(doc) for doc in documents]
        client = self._get_client()
        reranking_response = client.rerankings.create(
            model=self.model,
            query=query,
            top_n=self.top_n,
            texts=docs,
        )

        result_dicts = []
        for res in reranking_response.results:
            result_dicts.append({"index": res.index, "score": res.score})

        return result_dicts

    async def arerank(
        self, documents: Sequence[Union[Document, str]], query: str
    ) -> list[dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to
        the query.
        """
        docs = [self._document_to_str(doc) for doc in documents]
        client = self._get_aclient()
        reranking_response = await client.rerankings.create(
            model=self.model,
            query=query,
            top_n=self.top_n,
            texts=docs,
        )

        result_dicts = []
        for res in reranking_response.results:
            result_dicts.append({"index": res.index, "score": res.score})

        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using the Isaacus rerank API."""
        if not documents:
            return []

        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["score"] = res["score"]
            compressed.append(doc_copy)

        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents asynchronously using the Isaacus rerank API."""
        if not documents:
            return []

        compressed = []
        for res in await self.arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["score"] = res["score"]
            compressed.append(doc_copy)

        return compressed
