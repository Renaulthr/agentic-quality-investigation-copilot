from unittest.mock import patch


@patch(
    "api.main.get_dependency_status"
)
def test_readiness_success(
    mock_dependencies,
    client,
):
    mock_dependencies.return_value = {
        "quality_risk_api": {
            "status": "available",
            "detail": None,
        },
        "ollama": {
            "status": "available",
            "detail": None,
        },
        "mlflow": {
            "status": "available",
            "detail": None,
        },
        "checkpoint_database": {
            "status": "available",
            "detail": None,
        },
    }

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@patch(
    "api.main.get_dependency_status"
)
def test_readiness_failure(
    mock_dependencies,
    client,
):
    mock_dependencies.return_value = {
        "quality_risk_api": {
            "status": "available",
            "detail": None,
        },
        "ollama": {
            "status": "unavailable",
            "detail": "Connection refused",
        },
        "mlflow": {
            "status": "unavailable",
            "detail": "Connection refused",
        },
        "checkpoint_database": {
            "status": "available",
            "detail": None,
        },
    }

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"