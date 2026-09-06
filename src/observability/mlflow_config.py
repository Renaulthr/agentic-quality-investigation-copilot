import mlflow


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "agentic-quality-investigation"


def configure_mlflow():
    """
    Configure MLflow tracking and LangGraph tracing.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    # LangGraph / LangChain automatic tracing
    mlflow.langchain.autolog()

    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment": MLFLOW_EXPERIMENT,
    }

