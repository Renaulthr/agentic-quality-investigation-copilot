import pandas as pd
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

COMPLAINT_FILE = (
    PROJECT_ROOT
    / "data"
    / "complaints"
    / "customer_complaints.csv"
)


def get_complaint(
    complaint_id: str,
):
    """
    Retrieve one complaint record
    using Complaint_ID.
    """

    if not complaint_id.strip():
        raise ValueError(
            "complaint_id cannot be empty."
        )

    df = pd.read_csv(
        COMPLAINT_FILE
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
                "Complaint not found.",
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

    result = get_complaint(
        "CMP-001"
    )

    print(result)