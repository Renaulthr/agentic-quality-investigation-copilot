import time

import os

os.environ["MLFLOW_TRACKING_URI"] = (
    "http://127.0.0.1:5000"
)
from pathlib import Path

import pandas as pd

from src.tools.complaint_tool import (
    get_complaint,
)

from src.tools.traceability_tool import (
    get_traceability,
)

from src.tools.quality_risk_tool import (
    predict_quality_risk,
)

from src.tools.historical_case_tool import (
    search_historical_cases,
)

from src.rag.retriever import (
    retrieve_documents,
)

from src.llm.ollama_llm import (
    OllamaLLM,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "performance_benchmark.csv"
)


def measure(
    name: str,
    function,
    *args,
    **kwargs,
):
    """
    Measure execution time of one component.
    """

    start = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "component": name,
        "latency_seconds": round(
            elapsed,
            4,
        ),
        "result": result,
    }


def benchmark_case(
    complaint_id: str,
) -> list:

    measurements = []

    # -----------------------------------------
    # Complaint lookup
    # -----------------------------------------

    complaint_test = measure(
        "complaint_lookup",
        get_complaint,
        complaint_id,
    )

    measurements.append(
        complaint_test
    )

    complaint = (
        complaint_test[
            "result"
        ]
    )

    # -----------------------------------------
    # Traceability lookup
    # -----------------------------------------

    measurements.append(
        measure(
            "traceability_lookup",
            get_traceability,
            complaint_id,
        )
    )

    # -----------------------------------------
    # Project 1 ML API
    # -----------------------------------------

    measurements.append(
        measure(
            "quality_risk_api",
            predict_quality_risk,
            complaint_id,
        )
    )

    # -----------------------------------------
    # Historical search
    # -----------------------------------------

    issue = (
        complaint
        .get(
            "data",
            {},
        )
        .get(
            "Issue",
            "",
        )
    )

    measurements.append(
        measure(
            "historical_case_search",
            search_historical_cases,
            issue,
        )
    )

    # -----------------------------------------
    # General RAG retrieval
    # -----------------------------------------

    measurements.append(
        measure(
            "rag_retrieval",
            retrieve_documents,
            issue,
            4,
        )
    )

    # -----------------------------------------
    # Local Ollama test
    # -----------------------------------------

    llm = OllamaLLM(
        model="llama3.2"
    )

    benchmark_prompt = f"""
You are a manufacturing quality assistant.

Summarize the following complaint in no more
than three concise bullet points.

Complaint ID: {complaint_id}
Issue: {issue}
"""

    measurements.append(
        measure(
            "ollama_generation",
            llm.generate,
            benchmark_prompt,
        )
    )

    return measurements


def main():

    complaint_ids = [
        "CMP-001",
        "CMP-002",
        "CMP-003",
        "CMP-004",
        "CMP-005",
    ]

    rows = []

    print(
        "\nCOMPONENT PERFORMANCE BENCHMARK"
    )

    print(
        "=" * 70
    )

    for complaint_id in complaint_ids:

        print(
            f"\nBenchmarking {complaint_id}..."
        )

        measurements = benchmark_case(
            complaint_id
        )

        for measurement in measurements:

            latency = measurement[
                "latency_seconds"
            ]

            component = measurement[
                "component"
            ]

            print(
                f"{component:<28} "
                f"{latency:>8.2f} s"
            )

            rows.append(
                {
                    "complaint_id":
                        complaint_id,

                    "component":
                        component,

                    "latency_seconds":
                        latency,
                }
            )

    dataframe = pd.DataFrame(
        rows
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        REPORT_FILE,
        index=False,
    )

    summary = (
        dataframe
        .groupby(
            "component"
        )[
            "latency_seconds"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
            ]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AVERAGE COMPONENT LATENCY"
    )

    print(
        "=" * 70
    )

    print(
        summary.round(2)
    )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()