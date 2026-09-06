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


def collect_investigation_evidence(
    complaint_id: str,
):
    """
    Collect deterministic and retrieved evidence
    required for quality investigation.
    """

    complaint = get_complaint(
        complaint_id
    )

    traceability = get_traceability(
        complaint_id
    )

    quality_risk = predict_quality_risk(
        complaint_id
    )

    if complaint.get("found"):

        issue = complaint[
            "data"
        ].get(
            "Issue",
            ""
        )

        historical_cases = (
            search_historical_cases(
                issue,
                top_k=3,
            )
        )

    else:

        historical_cases = {
            "query": "",
            "matches": [],
        }

    return {
        "complaint_id":
            complaint_id,

        "complaint":
            complaint,

        "traceability":
            traceability,

        "quality_risk":
            quality_risk,

        "historical_cases":
            historical_cases,
    }


if __name__ == "__main__":

    result = (
        collect_investigation_evidence(
            "CMP-004"
        )
    )

    print(
        "\nINVESTIGATION EVIDENCE"
    )

    print(
        "-" * 50
    )

    print(result)