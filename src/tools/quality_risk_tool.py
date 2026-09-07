import os
from typing import Any, Dict

import mlflow
import requests


QUALITY_RISK_API_URL = os.getenv(
    "QUALITY_RISK_API_URL",
    "http://127.0.0.1:8000/predict",
)


@mlflow.trace(
    name="quality_risk_api",
    span_type="TOOL",
)
def predict_quality_risk(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call the Project 1 manufacturing-quality API
    and return the prediction response.
    """

    try:
        response = requests.post(
            QUALITY_RISK_API_URL,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        return {
            "success": True,
            "data": response.json(),
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "data": None,
            "error": "Quality risk API request timed out.",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "data": None,
            "error": "Quality risk API is unavailable.",
        }

    except requests.exceptions.HTTPError as error:
        return {
            "success": False,
            "data": None,
            "error": (
                "Quality risk API returned an HTTP error: "
                f"{error}"
            ),
        }

    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "data": None,
            "error": (
                "Quality risk API request failed: "
                f"{error}"
            ),
        }