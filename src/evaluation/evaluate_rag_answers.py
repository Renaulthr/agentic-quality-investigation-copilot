import json
import time
from pathlib import Path

import pandas as pd

from src.rag.rag_service import RAGService
from src.evaluation.grounding_checks import (contains_unsupported_claim,
                                             is_refusal,)

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
    / "rag_answer_evaluation.csv"
)

INSUFFICIENT_EVIDENCE_PHRASE = (
    "insufficient evidence"
)


def load_evaluation_data():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def keyword_coverage(
    answer: str,
    expected_keywords: list[str],
):
    """
    Fraction of expected keywords found in the answer.
    """

    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()

    matches = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    return (
        matches
        / len(expected_keywords)
    )


def evaluate_case(
    rag_service,
    item,
):

    question = item[
        "question"
    ]

    should_answer = item[
        "should_answer"
    ]

    expected_source = item[
        "expected_source"
    ]

    expected_keywords = item[
        "expected_keywords"
    ]

    start_time = (
        time.perf_counter()
    )

    result = rag_service.answer(
        question
    )

    latency_seconds = (
        time.perf_counter()
        - start_time
    )

    answer = result[
        "answer"
    ]

    retrieved_sources = [
        source["source"]
        for source in result[
            "sources"
        ]
    ]

    refused = is_refusal(
        answer
    )

    if should_answer:
        correct_behavior = (
            not refused
        )
    else:
        correct_behavior = (
            refused
        )

    source_correct = (
        expected_source
        in retrieved_sources
        if expected_source
        else refused
    )

    coverage = keyword_coverage(
        answer,
        expected_keywords,
    )

    return {
        "question":
            question,

        "should_answer":
            should_answer,

        "answer":
            answer,

        "refused":
            refused,

        "correct_behavior":
            int(
                correct_behavior
            ),

        "unsupported_claim":
            int(
                contains_unsupported_claim(
                    answer,
                    should_answer,
                )
            ), 
            
        "expected_source":
            expected_source,

        "retrieved_sources":
            " | ".join(
                retrieved_sources
            ),

        "source_correct":
            int(
                source_correct
            ),

        "keyword_coverage":
            coverage,

        "latency_seconds":
            latency_seconds,
    }


def main():

    evaluation_data = (
        load_evaluation_data()
    )

    rag_service = RAGService(
        top_k=3
    )

    rows = []

    print(
        "\nRunning RAG evaluation..."
    )

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):

        print(
            f"Evaluating "
            f"{index}/"
            f"{len(evaluation_data)}"
        )

        result = evaluate_case(
            rag_service,
            item,
        )

        rows.append(
            result
        )

    results_df = pd.DataFrame(
        rows
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_FILE,
        index=False,
    )
    print("\nFAILED BEHAVIOR CASES")
    print("-" * 60)

    failed_behavior = results_df[
        results_df["correct_behavior"] == 0
    ]

    for _, row in failed_behavior.iterrows():

        print(
            f"\nQuestion: {row['question']}"
        )

        print(
            f"Should Answer: {row['should_answer']}"
        )

        print(
            f"Refused: {row['refused']}"
        )

        print(
            f"Answer:\n{row['answer']}"
        )

    print("\nSOURCE RETRIEVAL FAILURES")
    print("-" * 60)

    failed_sources = results_df[
        results_df["source_correct"] == 0
    ]

    for _, row in failed_sources.iterrows():

        print(
            f"\nQuestion: {row['question']}"
        )

        print(
            f"Expected: {row['expected_source']}"
        )

        print(
            f"Retrieved: {row['retrieved_sources']}"
        )

    print("\nLOW KEYWORD COVERAGE")
    print("-" * 60)

    low_coverage = results_df[
        results_df["keyword_coverage"] < 0.67
    ]

    for _, row in low_coverage.iterrows():

        print(
            f"\nQuestion: {row['question']}"
        )

        print(
            f"Coverage: {row['keyword_coverage']:.3f}"
        )

        print(
            f"Answer:\n{row['answer']}"
        )
        
    behavior_accuracy = (
        results_df[
            "correct_behavior"
        ].mean()
    )

    source_accuracy = (
        results_df[
            "source_correct"
        ].mean()
    )

    average_keyword_coverage = (
        results_df[
            "keyword_coverage"
        ].mean()
    )

    average_latency = (
        results_df[
            "latency_seconds"
        ].mean()
    )

    unsupported_claim_rate = (
        results_df[
            "unsupported_claim"
        ].mean()
    )

    print(
        "\nRAG Evaluation Summary"
    )

    print(
        "-" * 40
    )

    print(
        f"Behavior Accuracy: "
        f"{behavior_accuracy:.3f}"
    )

    print(
        f"Source Accuracy: "
        f"{source_accuracy:.3f}"
    )

    print(
        f"Keyword Coverage: "
        f"{average_keyword_coverage:.3f}"
    )

    print(
        f"Unsupported Claim Rate: "
        f"{unsupported_claim_rate:.3f}"
    )

    print(
        f"Average Latency: "
        f"{average_latency:.2f}s"
    )

    print(
        "\nReport saved to:"
    )

    print(
        REPORT_FILE
    )

    print(
        results_df[
            [
                "question",
                "latency_seconds",
            ]
        ].sort_values(
            "latency_seconds",
            ascending=False,
        )
    )


if __name__ == "__main__":
    main()