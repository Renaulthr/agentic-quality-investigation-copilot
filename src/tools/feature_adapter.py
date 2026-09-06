from src.tools.complaint_tool import get_complaint
from src.tools.traceability_tool import get_traceability


REQUIRED_MODEL_FEATURES = [
    "Shift",
    "Model",
    "Machine_ID",
    "Temperature",
    "Pressure",
    "Cycle_Time",
    "Assembly_Force",
    "Ring_Gap",
    "Ring_Presence",
    "Circlip_Presence",
    "Oil_Ring_Presence",
]


def build_prediction_payload(
    complaint_id: str,
) -> dict:
    """
    Build the exact request body expected
    by Project 1 /predict endpoint.
    """

    complaint_result = get_complaint(
        complaint_id
    )

    traceability_result = get_traceability(
        complaint_id
    )

    if not complaint_result["found"]:
        raise ValueError(
            f"Complaint not found: {complaint_id}"
        )

    if not traceability_result["found"]:
        raise ValueError(
            f"Traceability not found: {complaint_id}"
        )

    complaint = complaint_result["data"]
    trace = traceability_result["data"]

    payload = {
        "Shift":
            str(trace["Shift"]),

        "Model":
            str(complaint["Model"]),

        "Machine_ID":
            str(trace["Machine_ID"]),

        "Temperature":
            float(trace["Temperature"]),

        "Pressure":
            float(trace["Pressure"]),

        "Cycle_Time":
            float(trace["Cycle_Time"]),

        "Assembly_Force":
            float(trace["Assembly_Force"]),

        "Ring_Gap":
            float(trace["Ring_Gap"]),

        "Ring_Presence":
            int(trace["Ring_Presence"]),

        "Circlip_Presence":
            int(trace["Circlip_Presence"]),

        "Oil_Ring_Presence":
            int(trace["Oil_Ring_Presence"]),
    }

    missing_features = [
        feature
        for feature in REQUIRED_MODEL_FEATURES
        if feature not in payload
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    return payload


if __name__ == "__main__":

    payload = build_prediction_payload(
        "CMP-001"
    )

    print("\nPROJECT 1 PREDICTION PAYLOAD")
    print("-" * 50)

    for key, value in payload.items():
        print(
            f"{key}: {value}"
        )