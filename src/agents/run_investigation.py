import mlflow

from langgraph.types import Command

from src.agents.investigation_graph import (
    build_investigation_graph,
)

from src.observability.mlflow_config import (
    configure_mlflow,
)

from src.tools.complaint_tool import (
    get_complaint,
)


@mlflow.trace(
    name="quality_investigation",
    span_type="AGENT",
)
def start_investigation(
    graph,
    complaint_id: str,
    config: dict,
):
    """
    Start the investigation workflow.
    """

    mlflow.update_current_trace(
        tags={
            "project":
                "agentic-quality-investigation-copilot",

            "complaint_id":
                complaint_id,

            "workflow_type":
                "manufacturing_quality_rca",

            "llm_provider":
                "ollama",

            "llm_model":
                "llama3.2",
        }
    )

    return graph.invoke(
        {
            "complaint_id":
                complaint_id,

            "errors": [],
        },
        config=config,
    )


def resume_investigation(
    graph,
    config: dict,
    approved: bool,
    comment: str,
):
    """
    Resume graph after human approval/rejection.
    """

    return graph.invoke(
        Command(
            resume={
                "approved":
                    approved,

                "comment":
                    comment,
            }
        ),
        config=config,
    )


def main():

    print(
        "\nQUALITY INVESTIGATION COPILOT"
    )

    print(
        "=" * 60
    )

    # -------------------------------------------------
    # 1. Configure MLflow
    # -------------------------------------------------

    try:

        configure_mlflow()

        print(
            "MLflow configuration: OK"
        )

    except Exception as error:

        print(
            f"\nMLflow configuration failed: {error}"
        )

        return

    # -------------------------------------------------
    # 2. Build graph
    # -------------------------------------------------

    try:

        graph = (
            build_investigation_graph()
        )

        print(
            "Investigation graph: OK"
        )

    except Exception as error:

        print(
            f"\nGraph creation failed: {error}"
        )

        return

    # -------------------------------------------------
    # 3. Ask for complaint ID
    # -------------------------------------------------

    complaint_id = input(
        "\nEnter Complaint ID "
        "(example CMP-004): "
    ).strip().upper()

    if not complaint_id:

        print(
            "\nERROR: Complaint ID cannot be empty."
        )

        return

    # -------------------------------------------------
    # 4. Validate complaint before running graph
    # -------------------------------------------------

    try:

        complaint_check = get_complaint(
            complaint_id
        )

    except Exception as error:

        print(
            f"\nComplaint lookup failed: {error}"
        )

        return

    if not complaint_check.get(
        "found",
        False,
    ):

        print(
            f"\nERROR: Complaint '{complaint_id}' "
            "was not found."
        )

        print(
            "\nAvailable synthetic complaints:"
        )

        print(
            "CMP-001, CMP-002, CMP-003, "
            "CMP-004, CMP-005"
        )

        return

    print(
        f"\nComplaint found: {complaint_id}"
    )

    issue = (
        complaint_check
        .get(
            "data",
            {}
        )
        .get(
            "Issue",
            "Unknown"
        )
    )

    print(
        f"Issue: {issue}"
    )

    # -------------------------------------------------
    # 5. LangGraph thread
    # -------------------------------------------------

    config = {
        "configurable": {
            "thread_id":
                f"investigation-{complaint_id}"
        }
    }

    print(
        "\nSTARTING INVESTIGATION"
    )

    print(
        "-" * 60
    )

    print(
        "Collecting complaint, traceability, "
        "ML and RAG evidence..."
    )

    # -------------------------------------------------
    # 6. Execute graph
    # -------------------------------------------------

    try:

        result = start_investigation(
            graph=graph,
            complaint_id=complaint_id,
            config=config,
        )

    except Exception as error:

        print(
            "\nINVESTIGATION FAILED"
        )

        print(
            "-" * 60
        )

        print(
            f"Error: {error}"
        )

        return

    # -------------------------------------------------
    # 7. Check errors returned inside state
    # -------------------------------------------------

    errors = result.get(
        "errors",
        [],
    )

    if errors:

        print(
            "\nWORKFLOW ERRORS"
        )

        print(
            "-" * 60
        )

        for error in errors:

            print(
                f"- {error}"
            )

    # -------------------------------------------------
    # 8. Show investigation summary
    # -------------------------------------------------

    summary = result.get(
        "investigation_summary"
    )

    if summary:

        print(
            "\nINVESTIGATION SUMMARY"
        )

        print(
            "-" * 60
        )

        print(
            summary
        )

    # -------------------------------------------------
    # 9. Show RCA hypothesis
    # -------------------------------------------------

    hypothesis = result.get(
        "root_cause_hypothesis"
    )

    if hypothesis:

        print(
            "\nRCA HYPOTHESIS"
        )

        print(
            "-" * 60
        )

        print(
            hypothesis
        )

    # -------------------------------------------------
    # 10. Evidence confidence
    # -------------------------------------------------

    evidence_strength = result.get(
        "evidence_strength"
    )

    confidence = result.get(
        "confidence"
    )

    if evidence_strength:

        print(
            "\nEVIDENCE ASSESSMENT"
        )

        print(
            "-" * 60
        )

        print(
            f"Evidence Strength: "
            f"{evidence_strength}"
        )

        print(
            f"Confidence: "
            f"{confidence}"
        )

    # -------------------------------------------------
    # 11. Detect human-review interrupt
    # -------------------------------------------------

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if not interrupts:

        print(
            "\nNo human-review interrupt was generated."
        )

        final_report = result.get(
            "final_report"
        )

        if final_report:

            print(
                "\nFINAL INVESTIGATION REPORT"
            )

            print(
                "-" * 60
            )

            print(
                final_report
            )

        else:

            print(
                "\nWorkflow completed without "
                "a final report."
            )

            print(
                "\nReturned state keys:"
            )

            print(
                list(result.keys())
            )

        return

    # -------------------------------------------------
    # 12. Human approval
    # -------------------------------------------------

    print(
        "\nHUMAN REVIEW REQUIRED"
    )

    print(
        "=" * 60
    )

    for item in interrupts:

        print(
            item.value
        )

    while True:

        user_input = input(
            "\nApprove hypothesis? "
            "(yes/no): "
        ).strip().lower()

        if user_input in {
            "yes",
            "y",
        }:

            approved = True
            break

        if user_input in {
            "no",
            "n",
        }:

            approved = False
            break

        print(
            "Please enter yes or no."
        )

    comment = input(
        "Review comment: "
    ).strip()

    # -------------------------------------------------
    # 13. Resume same graph thread
    # -------------------------------------------------

    print(
        "\nResuming investigation..."
    )

    try:

        final_result = (
            resume_investigation(
                graph=graph,
                config=config,
                approved=approved,
                comment=comment,
            )
        )

    except Exception as error:

        print(
            "\nFailed to resume investigation."
        )

        print(
            f"Error: {error}"
        )

        return

    # -------------------------------------------------
    # 14. Final report
    # -------------------------------------------------

    print(
        "\nFINAL INVESTIGATION REPORT"
    )

    print(
        "=" * 60
    )

    final_report = (
        final_result.get(
            "final_report"
        )
    )

    if final_report:

        print(
            final_report
        )

    else:

        print(
            "No final report was generated."
        )

        print(
            "\nReturned state keys:"
        )

        print(
            list(
                final_result.keys()
            )
        )


if __name__ == "__main__":
    main()