import requests
import mlflow

from src.tools.feature_adapter import (
    build_prediction_payload,
)


DEFAULT_API_URL = (
    "http://localhost:8000/predict"
)

@mlflow.trace(
    name="quality_risk_ml_api",
    span_type="TOOL",
)

def predict_quality_risk(
    complaint_id: str,
    api_url: str = DEFAULT_API_URL,
) -> dict:
    """
    Build process features for a complaint
    and call Project 1 ML prediction API.
    """

    if not complaint_id.strip():
        raise ValueError(
            "complaint_id cannot be empty."
        )

    try:
        payload = build_prediction_payload(
            complaint_id
        )

    except ValueError as error:

        return {
            "success": False,
            "complaint_id": complaint_id,
            "stage": "feature_adapter",
            "error": str(error),
        }

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        prediction = response.json()

        return {
            "success": True,
            "complaint_id": complaint_id,
            "input_features": payload,
            "prediction": prediction,
        }

    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "complaint_id": complaint_id,
            "stage": "api_connection",
            "error":
                "Quality risk API is unavailable.",
        }

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "complaint_id": complaint_id,
            "stage": "api_timeout",
            "error":
                "Quality risk API request timed out.",
        }

    except requests.exceptions.HTTPError as error:

        return {
            "success": False,
            "complaint_id": complaint_id,
            "stage": "api_response",
            "status_code":
                response.status_code,
            "error":
                response.text,
        }

    except requests.exceptions.RequestException as error:

        return {
            "success": False,
            "complaint_id": complaint_id,
            "stage": "api_request",
            "error": str(error),
        }


if __name__ == "__main__":

    result = predict_quality_risk(
        "CMP-004"
    )

    print("\nQUALITY RISK RESULT")
    print("-" * 50)

    print(result)