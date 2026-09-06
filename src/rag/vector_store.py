from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from src.rag.embeddings import (
    get_embeddings,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)

COLLECTION_NAME = (
    "quality_knowledge"
)


@lru_cache(maxsize=1)
def get_vector_store():
    """
    Load the persistent Chroma store once
    per Python process.
    """

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(
            VECTOR_STORE_PATH
        ),
    )