from src.rag.vector_store import (
    get_vector_store,
)


def retrieve_documents(
    query: str,
    top_k: int = 3,
):
    """
    Retrieve the most relevant quality
    knowledge chunks.
    """

    vector_store = (
        get_vector_store()
    )

    results = (
        vector_store
        .similarity_search_with_score(
            query,
            k=top_k,
        )
    )

    retrieved = []

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        retrieved.append(
            {
                "rank":
                    rank,

                "source":
                    document.metadata.get(
                        "source",
                        "",
                    ),

                "chunk_id":
                    document.metadata.get(
                        "chunk_id",
                        "",
                    ),

                "distance":
                    float(score),

                "content":
                    document.page_content,
            }
        )

    return retrieved