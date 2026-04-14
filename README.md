# langchain-isaacus

This package contains the LangChain integration for the [Isaacus](https://isaacus.com) legal API.

## Installation

```bash
pip install -U langchain-isaacus
```

And you should configure credentials by setting the following environment variables:

- `ISAACUS_API_KEY`: Your Isaacus API key. You can get your API key by following the [Isaacus API quickstart guide](https://docs.isaacus.com/quickstart#1-set-up-your-account) and, in particular, joining the [Isaacus Platform](https://platform.isaacus.com/accounts/signup/).

## Embeddings

`IsaacusEmbeddings` class exposes embeddings from Isaacus.

```python
from langchain_isaacus import IsaacusEmbeddings

embeddings = IsaacusEmbeddings(model="kanon-2-embedder")
embeddings.embed_query("What is the meaning of life?")
```

## Reranking

`IsaacusRerank` class exposes reranking from Isaacus. 

```python
from langchain_isaacus import IsaacusRerank
from langchain_core.documents import Document

rerankings = IsaacusRerank(model="kanon-2-reranker")
embeddings.compress_documents(
    documents=[Document("The meaning of life is ...")],
    query="What is the meaning of life?"
)
```