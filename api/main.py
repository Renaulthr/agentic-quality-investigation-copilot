from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.investigations import (
    router as investigation_router,
)

from src.observability.mlflow_config import (
    configure_mlflow,
)

from src.schemas.api_schemas import (
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        configure_mlflow()

        print(
            "MLflow observability: enabled"
        )

    except Exception as error:

        print(
            "MLflow observability: unavailable"
        )

        print(
            f"Reason: {error}"
        )

    yield


app = FastAPI(
    title=(
        "Agentic Quality "
        "Investigation Copilot"
    ),
    description=(
        "Agentic AI system for "
        "manufacturing quality "
        "investigation using RAG, "
        "ML risk scoring, LangGraph "
        "and human-in-the-loop review."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(
    investigation_router
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():

    return {
        "status": "healthy",
        "service":
            "agentic-quality-"
            "investigation-copilot",
        "version": "1.0.0",
    }