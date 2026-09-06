from src.rag.retriever import (
    retrieve_documents,
)


def build_context(
    query: str,
    top_k: int = 3,
):

    results = retrieve_documents(
        query=query,
        top_k=top_k,
    )

    context_parts = []
    sources = []

    for index, item in enumerate(
        results,
        start=1,
    ):

        source = item.get(
            "source",
            "",
        )

        chunk_id = item.get(
            "chunk_id",
            "",
        )

        distance = item.get(
            "distance",
        )

        content = item.get(
            "content",
            "",
        )

        context_parts.append(
            f"""
[Evidence {index}]
Source: {source}
Chunk: {chunk_id}
Content:
{content}
""".strip()
        )

        sources.append(
            {
                "rank": index,
                "source": source,
                "chunk_id": chunk_id,
                "distance": distance,
            }
        )

    context = "\n\n".join(
        context_parts
    )

    return context, sources