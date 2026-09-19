from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    complaint_id: str = Field(
        ...,
        examples=["CMP-004"],
    )


class HumanReviewRequest(BaseModel):
    approved: bool
    comment: Optional[str] = None


class InvestigationStartResponse(BaseModel):
    investigation_id: str
    complaint_id: str
    status: str

    investigation_summary: Optional[str] = None
    root_cause_hypothesis: Optional[str] = None

    evidence_strength: Optional[str] = None
    confidence: Optional[str] = None

    requires_human_review: bool = True

    errors: List[str] = Field(
        default_factory=list
    )


class InvestigationReviewResponse(BaseModel):
    investigation_id: str
    complaint_id: str
    status: str

    human_approved: bool
    human_comment: Optional[str] = None

    final_report: Optional[str] = None

    errors: List[str] = Field(
        default_factory=list
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ErrorResponse(BaseModel):
    detail: str

from typing import Dict, Optional

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    status: str
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    status: str
    service: str
    dependencies: Dict[str, DependencyStatus]    