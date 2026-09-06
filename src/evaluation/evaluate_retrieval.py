import json
from pathlib import Path

import pandas as pd

from src.rag.retriever import (
    retrieve_documents,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_evaluation.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "retrieval_evaluation.csv"
)


def load_evaluation_set():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def evaluate_retrieval(
    top_k: int = 3,
):

    evaluation_data = (
        load_evaluation_set()
    )

    rows = []

    for item in evaluation_data:

        question = item[
            "question"
        ]

        expected_source = item[
            "expected_source"
        ]

        results = retrieve_documents(
            question,
            top_k=top_k,
        )

        retrieved_sources = [
            document.metadata[
                "source"
            ]
            for document, _
            in results
        ]

        source_hit = (
            expected_source
            in retrieved_sources
        )

        rank = None

        if source_hit:

            rank = (
                retrieved_sources
                .index(
                    expected_source
                )
                + 1
            )

        reciprocal_rank = (
            1 / rank
            if rank
            else 0
        )

        rows.append(
            {
                "question":
                    question,

                "expected_source":
                    expected_source,

                "retrieved_sources":
                    " | ".join(
                        retrieved_sources
                    ),

                "hit_at_k":
                    int(source_hit),

                "rank":
                    rank,

                "reciprocal_rank":
                    reciprocal_rank,
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    recall_at_k = (
        results_df[
            "hit_at_k"
        ].mean()
    )

    mrr = (
        results_df[
            "reciprocal_rank"
        ].mean()
    )

    return (
        results_df,
        recall_at_k,
        mrr,
    )


def main():

    (
        results_df,
        recall_at_k,
        mrr,
    ) = evaluate_retrieval(
        top_k=3
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_FILE,
        index=False,
    )

    print(
        "\nRetrieval Evaluation"
    )

    print(
        "-" * 40
    )

    print(
        results_df[
            [
                "question",
                "expected_source",
                "rank",
                "hit_at_k",
            ]
        ]
    )

    print(
        f"\nRecall@3: "
        f"{recall_at_k:.3f}"
    )

    print(
        f"MRR: "
        f"{mrr:.3f}"
    )

    print(
        "\nSaved report:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":
    main()