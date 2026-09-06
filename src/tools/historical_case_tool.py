import mlflow

from src.rag.retriever import (
    retrieve_documents,
)


@mlflow.trace(
    name="historical_case_search",
    span_type="RETRIEVER",
)
def search_historical_cases(
    query: str,
    top_k: int = 3,
) -> dict:
    """
    Search historical quality investigation cases
    relevant to the supplied issue.
    """

    results = retrieve_documents(
        query=query,
        top_k=top_k,
    )

    matches = []

    for item in results:

        if (
            item.get("source")
            != "historical_8d_cases.md"
        ):
            continue

        matches.append(
            {
                "source":
                    item.get(
                        "source",
                        "",
                    ),

                "chunk_id":
                    item.get(
                        "chunk_id",
                        "",
                    ),

                "distance":
                    item.get(
                        "distance",
                    ),

                "content":
                    item.get(
                        "content",
                        "",
                    ),
            }
        )

    return {
        "query":
            query,

        "matches":
            matches,
    }