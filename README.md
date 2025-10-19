# langchain-isaacus

This package contains the LangChain integration with Isaacus

## Installation

```bash
pip install -U langchain-isaacus
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatIsaacus` class exposes chat models from Isaacus.

```python
from langchain_isaacus import ChatIsaacus

llm = ChatIsaacus()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`IsaacusEmbeddings` class exposes embeddings from Isaacus.

```python
from langchain_isaacus import IsaacusEmbeddings

embeddings = IsaacusEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs

`IsaacusLLM` class exposes LLMs from Isaacus.

```python
from langchain_isaacus import IsaacusLLM

llm = IsaacusLLM()
llm.invoke("The meaning of life is")
```
