from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status

from api.routes.investigations import (
    router as investigation_router,
)
from src.observability.mlflow_config import (
    configure_mlflow,
)
from src.health.dependency_checks import (
    get_dependency_status,
)
from src.schemas.api_schemas import (
    HealthResponse,
    ReadinessResponse,
)

from src.logging_config.logger import (
    configure_logging,
    get_logger,
)

configure_logging()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Application starting",
        extra={
            "workflow_stage": "startup",
            "status": "starting",
        },
    )

    try:
        configure_mlflow()

        logger.info(
            "MLflow observability enabled",
            extra={
                "workflow_stage": "startup",
                "status": "available",
            },
        )

    except Exception as error:
        logger.warning(
            "MLflow observability unavailable",
            extra={
                "workflow_stage": "startup",
                "status": "unavailable",
                "error": str(error),
            },
        )

    yield

    logger.info(
        "Application shutting down",
        extra={
            "workflow_stage": "shutdown",
            "status": "stopping",
        },
    )
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
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
)
def readiness_check(
    response: Response,
):
    dependencies = get_dependency_status()

    critical_dependencies = [
        dependencies["quality_risk_api"],
        dependencies["ollama"],
        dependencies["checkpoint_database"],
    ]

    critical_ready = all(
        item["status"] == "available"
        for item in critical_dependencies
    )

    if not critical_ready:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return {
        "status": (
            "ready"
            if critical_ready
            else "not_ready"
        ),
        "service": (
            "agentic-quality-investigation-copilot"
        ),
        "dependencies": dependencies,
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():
    return {
        "status": "healthy",
        "service": (
            "agentic-quality-investigation-copilot"
        ),
        "version": "1.0.0",
    }