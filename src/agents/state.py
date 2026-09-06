from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class InvestigationState(TypedDict, total=False):

    complaint_id: str

    complaint: Dict[str, Any]
    traceability: Dict[str, Any]
    quality_risk: Dict[str, Any]
    historical_cases: Dict[str, Any]
    rag_evidence: List[Dict[str, Any]]

    investigation_summary: str

    root_cause_hypothesis: str

    evidence_strength: str
    confidence: str

    requires_human_review: bool

    human_approved: Optional[bool]
    human_comment: Optional[str]

    final_report: str

    errors: List[str]