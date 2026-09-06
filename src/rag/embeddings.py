from functools import lru_cache

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)


EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load the embedding model once per process.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )