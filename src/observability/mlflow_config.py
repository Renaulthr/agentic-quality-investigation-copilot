import os

import mlflow


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "agentic-quality-investigation",
)


def configure_mlflow():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    mlflow.langchain.autolog()

    return {
        "tracking_uri":
            MLFLOW_TRACKING_URI,

        "experiment":
            MLFLOW_EXPERIMENT,
    }