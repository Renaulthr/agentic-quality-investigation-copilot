from fastapi import APIRouter, HTTPException

from src.schemas.api_schemas import (
    ErrorResponse,
    HumanReviewRequest,
    InvestigationRequest,
    InvestigationReviewResponse,
    InvestigationStartResponse,
)

from src.services.investigation_service import (
    investigation_service,
)


router = APIRouter(
    prefix="/investigations",
    tags=["Investigations"],
)


@router.post(
    "",
    response_model=InvestigationStartResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def start_investigation(
    request: InvestigationRequest,
):
    try:
        result = (
            investigation_service
            .start_investigation(
                request.complaint_id
            )
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@router.post(
    "/{investigation_id}/review",
    response_model=InvestigationReviewResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def review_investigation(
    investigation_id: str,
    request: HumanReviewRequest,
):
    try:
        result = (
            investigation_service
            .review_investigation(
                investigation_id=
                    investigation_id,

                approved=
                    request.approved,

                comment=
                    request.comment,
            )
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error