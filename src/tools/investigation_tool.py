from src.tools.complaint_tool import (
    get_complaint,
)

from src.tools.traceability_tool import (
    get_traceability,
)


def get_investigation_data(
    complaint_id: str,
):
    """
    Collect deterministic investigation
    evidence for one complaint.
    """

    complaint = get_complaint(
        complaint_id
    )

    traceability = get_traceability(
        complaint_id
    )

    return {
        "complaint_id":
            complaint_id,

        "complaint":
            complaint,

        "traceability":
            traceability,
    }


if __name__ == "__main__":

    result = get_investigation_data(
        "CMP-001"
    )

    print(result)