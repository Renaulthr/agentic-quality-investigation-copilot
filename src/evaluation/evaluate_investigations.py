import json
import time
from pathlib import Path

import pandas as pd

from src.agents.investigation_graph import (
    build_investigation_graph,
)

from src.evaluation.investigation_checks import (
    check_expected_value,
    check_source_present,
    evaluate_rca_safety,
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
    / "investigation_evaluation.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "investigation_evaluation.csv"
)


def load_evaluation_cases():
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_historical_sources(
    historical_cases: dict,
) -> list:

    matches = historical_cases.get(
        "matches",
        [],
    )

    return [
        item.get(
            "source",
            ""
        )
        for item in matches
    ]


def evaluate_case(
    graph,
    case: dict,
) -> dict:

    complaint_id = case[
        "complaint_id"
    ]

    config = {
        "configurable": {
            "thread_id":
                f"evaluation-{complaint_id}"
        }
    }

    start_time = time.perf_counter()

    try:

        result = graph.invoke(
            {
                "complaint_id":
                    complaint_id,

                "errors": [],
            },
            config=config,
        )

        success = True
        error_message = ""

    except Exception as error:

        success = False
        error_message = str(
            error
        )

        result = {}

    latency_seconds = (
        time.perf_counter()
        - start_time
    )

    complaint = result.get(
        "complaint",
        {}
    ).get(
        "data",
        {}
    )

    traceability = result.get(
        "traceability",
        {}
    ).get(
        "data",
        {}
    )

    quality_prediction = result.get(
        "quality_risk",
        {}
    ).get(
        "prediction",
        {}
    )

    historical_cases = result.get(
        "historical_cases",
        {}
    )

    hypothesis = result.get(
        "root_cause_hypothesis",
        "",
    )

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    human_review_generated = (
        len(interrupts) > 0
    )

    # ----------------------------------------
    # Deterministic checks
    # ----------------------------------------

    issue_correct = True

    if "expected_issue" in case:

        issue_correct = (
            check_expected_value(
                complaint.get(
                    "Issue"
                ),
                case[
                    "expected_issue"
                ],
            )
        )

    final_inspection_correct = True

    if (
        "expected_final_inspection"
        in case
    ):

        final_inspection_correct = (
            check_expected_value(
                traceability.get(
                    "Final_Inspection"
                ),
                case[
                    "expected_final_inspection"
                ],
            )
        )

    ring_presence_correct = True

    if (
        "expected_ring_presence"
        in case
    ):

        ring_presence_correct = (
            check_expected_value(
                traceability.get(
                    "Ring_Presence"
                ),
                case[
                    "expected_ring_presence"
                ],
            )
        )

    circlip_presence_correct = True

    if (
        "expected_circlip_presence"
        in case
    ):

        circlip_presence_correct = (
            check_expected_value(
                traceability.get(
                    "Circlip_Presence"
                ),
                case[
                    "expected_circlip_presence"
                ],
            )
        )

    process_risk_correct = True

    if (
        "expected_process_risk"
        in case
    ):

        process_risk_correct = (
            check_expected_value(
                quality_prediction.get(
                    "quality_prediction"
                ),
                case[
                    "expected_process_risk"
                ],
            )
        )

    process_condition_correct = True

    if (
        "expected_process_condition"
        in case
    ):

        process_condition_correct = (
            check_expected_value(
                quality_prediction.get(
                    "process_condition"
                ),
                case[
                    "expected_process_condition"
                ],
            )
        )

    quality_state_correct = True

    if (
        "expected_quality_state"
        in case
    ):

        quality_state_correct = (
            check_expected_value(
                quality_prediction.get(
                    "quality_state"
                ),
                case[
                    "expected_quality_state"
                ],
            )
        )

    source_correct = True

    if (
        "expected_historical_source"
        in case
    ):

        historical_sources = (
            extract_historical_sources(
                historical_cases
            )
        )

        source_correct = (
            check_source_present(
                historical_sources,
                case[
                    "expected_historical_source"
                ],
            )
        )

    rca_safety = (
        evaluate_rca_safety(
            hypothesis=hypothesis,
            must_not_confirm_root_cause=
                case.get(
                    "must_not_confirm_root_cause",
                    True,
                ),
        )
    )

    expected_human_review = (
        case.get(
            "requires_human_review",
            True,
        )
    )

    human_gate_correct = (
        human_review_generated
        ==
        expected_human_review
    )

    checks = []

    if "expected_issue" in case:
        checks.append(
            issue_correct
        )

    if "expected_final_inspection" in case:
        checks.append(
            final_inspection_correct
        )

    if "expected_ring_presence" in case:
        checks.append(
            ring_presence_correct
        )

    if "expected_circlip_presence" in case:
        checks.append(
            circlip_presence_correct
        )

    if "expected_process_risk" in case:
        checks.append(
            process_risk_correct
        )

    if "expected_process_condition" in case:
        checks.append(
            process_condition_correct
        )

    if "expected_quality_state" in case:
        checks.append(
            quality_state_correct
        )

    if "expected_historical_source" in case:
        checks.append(
            source_correct
        )

    if "must_not_confirm_root_cause" in case:
        checks.append(
            rca_safety["passed"]
        )

    if "requires_human_review" in case:
        checks.append(
            human_gate_correct
        )

    passed_checks = sum(
        bool(value)
        for value in checks
    )

    case_score = (
        passed_checks / len(checks)
        if checks
        else 0.0
    )

    passed_checks = sum(
        bool(value)
        for value in checks
    )

    case_score = (
        passed_checks
        / len(checks)
    )

    return {
        "case_id":
            case[
                "case_id"
            ],

        "complaint_id":
            complaint_id,

        "execution_success":
            success,

        "issue_correct":
            issue_correct,

        "final_inspection_correct":
            final_inspection_correct,

        "ring_presence_correct":
            ring_presence_correct,

        "circlip_presence_correct":
            circlip_presence_correct,

        "process_risk_correct":
            process_risk_correct,

        "process_condition_correct":
            process_condition_correct,

        "quality_state_correct":
            quality_state_correct,

        "historical_source_correct":
            source_correct,

        "rca_safety_passed":
            rca_safety[
                "passed"
            ],

        "confirmed_root_cause_language":
            rca_safety[
                "confirmed_root_cause_language"
            ],

        "human_review_generated":
            human_review_generated,

        "human_gate_correct":
            human_gate_correct,

        "case_score":
            round(
                case_score,
                4,
            ),

        "latency_seconds":
            round(
                latency_seconds,
                2,
            ),

        "error":
            error_message,
    }


def main():

    cases = (
        load_evaluation_cases()
    )

    graph = (
        build_investigation_graph()
    )

    results = []

    print(
        "\nEND-TO-END AGENT EVALUATION"
    )

    print(
        "=" * 70
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            f"\n[{index}/{len(cases)}] "
            f"Evaluating "
            f"{case['complaint_id']}..."
        )

        result = evaluate_case(
            graph,
            case,
        )

        results.append(
            result
        )

        print(
            f"Score: "
            f"{result['case_score']:.2%}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']} s"
        )

        if result[
            "error"
        ]:

            print(
                f"Error: "
                f"{result['error']}"
            )

    dataframe = pd.DataFrame(
        results
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        REPORT_FILE,
        index=False,
    )

    # ----------------------------------------
    # KPI summary
    # ----------------------------------------

    completion_rate = (
        dataframe[
            "execution_success"
        ].mean()
    )

    grounding_safety_rate = (
        dataframe[
            "rca_safety_passed"
        ].mean()
    )

    human_gate_rate = (
        dataframe[
            "human_gate_correct"
        ].mean()
    )

    source_accuracy = (
        dataframe[
            "historical_source_correct"
        ].mean()
    )

    average_case_score = (
        dataframe[
            "case_score"
        ].mean()
    )

    average_latency = (
        dataframe[
            "latency_seconds"
        ].mean()
    )

    confirmed_claim_rate = (
        dataframe[
            "confirmed_root_cause_language"
        ].mean()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Investigation Completion Rate : "
        f"{completion_rate:.3f}"
    )

    print(
        f"Average Case Score            : "
        f"{average_case_score:.3f}"
    )

    print(
        f"RCA Safety Rate               : "
        f"{grounding_safety_rate:.3f}"
    )

    print(
        f"Human-Gate Compliance         : "
        f"{human_gate_rate:.3f}"
    )

    print(
        f"Historical Source Accuracy    : "
        f"{source_accuracy:.3f}"
    )

    print(
        f"Confirmed Root-Cause Rate     : "
        f"{confirmed_claim_rate:.3f}"
    )

    print(
        f"Average Investigation Latency : "
        f"{average_latency:.2f} s"
    )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()