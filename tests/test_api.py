from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert (
        body["service"]
        == "agentic-quality-investigation-copilot"
    )
    assert body["version"] == "1.0.0"


@patch(
    "api.routes.investigations."
    "investigation_service."
    "start_investigation"
)
def test_invalid_complaint(
    mock_start,
):

    mock_start.side_effect = ValueError(
        "Complaint ID 'CMP-999' was not found."
    )

    response = client.post(
        "/investigations",
        json={
            "complaint_id": "CMP-999"
        },
    )

    assert response.status_code == 400

    body = response.json()

    assert (
        "CMP-999"
        in body["detail"]
    )


@patch(
    "api.routes.investigations."
    "investigation_service."
    "start_investigation"
)
def test_start_investigation(
    mock_start,
):

    mock_start.return_value = {
        "investigation_id":
            "INV-TEST001",

        "complaint_id":
            "CMP-004",

        "status":
            "awaiting_human_review",

        "investigation_summary":
            "Synthetic investigation summary.",

        "root_cause_hypothesis":
            "Possible process-related cause.",

        "evidence_strength":
            "Strong",

        "confidence":
            "High",

        "requires_human_review":
            True,

        "errors":
            [],
    }

    response = client.post(
        "/investigations",
        json={
            "complaint_id": "CMP-004"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["investigation_id"]
        == "INV-TEST001"
    )

    assert (
        body["complaint_id"]
        == "CMP-004"
    )

    assert (
        body["status"]
        == "awaiting_human_review"
    )

    assert (
        body["requires_human_review"]
        is True
    )


@patch(
    "api.routes.investigations."
    "investigation_service."
    "review_investigation"
)
def test_approve_investigation(
    mock_review,
):

    mock_review.return_value = {
        "investigation_id":
            "INV-TEST001",

        "complaint_id":
            "CMP-004",

        "status":
            "completed",

        "human_approved":
            True,

        "human_comment":
            "Evidence reviewed and approved.",

        "final_report":
            "Synthetic final investigation report.",

        "errors":
            [],
    }

    response = client.post(
        "/investigations/"
        "INV-TEST001/review",

        json={
            "approved": True,
            "comment":
                "Evidence reviewed and approved.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "completed"
    )

    assert (
        body["human_approved"]
        is True
    )

    assert (
        body["final_report"]
        is not None
    )


@patch(
    "api.routes.investigations."
    "investigation_service."
    "review_investigation"
)
def test_reject_investigation(
    mock_review,
):

    mock_review.return_value = {
        "investigation_id":
            "INV-TEST002",

        "complaint_id":
            "CMP-001",

        "status":
            "rejected",

        "human_approved":
            False,

        "human_comment":
            "Additional verification required.",

        "final_report":
            "Further investigation required.",

        "errors":
            [],
    }

    response = client.post(
        "/investigations/"
        "INV-TEST002/review",

        json={
            "approved": False,
            "comment":
                "Additional verification required.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "rejected"
    )

    assert (
        body["human_approved"]
        is False
    )