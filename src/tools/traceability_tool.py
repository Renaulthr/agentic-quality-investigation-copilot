import pandas as pd
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TRACEABILITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "traceability"
    / "production_traceability.csv"
)


def get_traceability(
    complaint_id: str,
):
    """
    Retrieve production traceability
    linked to a complaint.
    """

    if not complaint_id.strip():
        raise ValueError(
            "complaint_id cannot be empty."
        )

    df = pd.read_csv(
        TRACEABILITY_FILE
    )

    match = df[
        df["Complaint_ID"]
        .str.upper()
        == complaint_id.upper()
    ]

    if match.empty:
        return {
            "found": False,
            "complaint_id": complaint_id,
            "message":
                "Traceability record not found.",
        }

    record = (
        match
        .iloc[0]
        .to_dict()
    )

    return {
        "found": True,
        "data": record,
    }


if __name__ == "__main__":

    result = get_traceability(
        "CMP-001"
    )

    print(result)