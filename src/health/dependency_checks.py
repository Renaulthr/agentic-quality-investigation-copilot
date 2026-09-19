import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

import requests


QUALITY_RISK_API_URL = os.getenv(
    "QUALITY_RISK_API_URL",
    "http://127.0.0.1:8000/predict",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

CHECKPOINT_DB_PATH = os.getenv(
    "CHECKPOINT_DB_PATH",
    "data/checkpoints/investigations.db",
)


def _check_http(
    url: str,
    timeout: float = 3.0,
) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            timeout=timeout,
        )

        if response.ok:
            return {
                "status": "available",
                "detail": None,
            }

        return {
            "status": "unavailable",
            "detail": f"HTTP {response.status_code}",
        }

    except requests.RequestException as error:
        return {
            "status": "unavailable",
            "detail": str(error),
        }


def check_quality_risk_api() -> Dict[str, Any]:
    base_url = QUALITY_RISK_API_URL.rsplit(
        "/predict",
        1,
    )[0]

    return _check_http(
        f"{base_url}/health"
    )


def check_ollama() -> Dict[str, Any]:
    return _check_http(
        f"{OLLAMA_BASE_URL}/api/tags"
    )


def check_mlflow() -> Dict[str, Any]:
    return _check_http(
        MLFLOW_TRACKING_URI
    )


def check_checkpoint_database() -> Dict[str, Any]:
    try:
        db_path = Path(CHECKPOINT_DB_PATH)

        db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            db_path,
            timeout=3,
        )

        connection.execute(
            "SELECT 1"
        )

        connection.close()

        return {
            "status": "available",
            "detail": None,
        }

    except sqlite3.Error as error:
        return {
            "status": "unavailable",
            "detail": str(error),
        }


def get_dependency_status() -> Dict[str, Dict[str, Any]]:
    return {
        "quality_risk_api": check_quality_risk_api(),
        "ollama": check_ollama(),
        "mlflow": check_mlflow(),
        "checkpoint_database": check_checkpoint_database(),
    }