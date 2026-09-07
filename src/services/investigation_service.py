from typing import Any, Dict
from uuid import uuid4

from langgraph.types import Command

from src.agents.investigation_graph import (
    build_investigation_graph,
)

from src.tools.complaint_tool import (
    get_complaint,
)


class InvestigationService:
    """
    Application service for starting and resuming
    manufacturing quality investigations.

    The LangGraph instance is created once and reused
    so that checkpoint state remains available during
    the application process lifetime.
    """

    def __init__(self) -> None:

        self.graph = (
            build_investigation_graph()
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    @staticmethod
    def _build_config(
        investigation_id: str,
    ) -> dict:
        """
        Build the LangGraph configuration.

        investigation_id is also used as the LangGraph
        thread_id so the same investigation can later
        be resumed after human review.
        """

        return {
            "configurable": {
                "thread_id":
                    investigation_id
            }
        }

    @staticmethod
    def _extract_interrupt(
        result: Dict[str, Any],
    ) -> bool:
        """
        Determine whether LangGraph paused for
        human review.
        """

        interrupts = result.get(
            "__interrupt__",
            [],
        )

        return bool(
            interrupts
        )

    # --------------------------------------------------
    # Start investigation
    # --------------------------------------------------

    def start_investigation(
        self,
        complaint_id: str,
    ) -> Dict[str, Any]:

        complaint_id = (
            complaint_id
            .strip()
            .upper()
        )

        if not complaint_id:

            raise ValueError(
                "Complaint ID cannot be empty."
            )

        # Validate complaint before launching
        # expensive RAG / ML / LLM processing.

        complaint = get_complaint(
            complaint_id
        )

        if not complaint.get(
            "found",
            False,
        ):

            raise ValueError(
                f"Complaint ID "
                f"'{complaint_id}' "
                f"was not found."
            )

        investigation_id = (
            f"INV-{uuid4().hex[:12].upper()}"
        )

        config = (
            self._build_config(
                investigation_id
            )
        )

        try:

            result = self.graph.invoke(
                {
                    "complaint_id":
                        complaint_id,

                    "errors":
                        [],
                },
                config=config,
            )

        except Exception as error:

            raise RuntimeError(
                "Investigation workflow failed: "
                f"{error}"
            ) from error

        human_review_required = (
            self._extract_interrupt(
                result
            )
        )

        if human_review_required:

            status = (
                "awaiting_human_review"
            )

        elif result.get(
            "final_report"
        ):

            status = "completed"

        else:

            status = "processing_incomplete"

        return {
            "investigation_id":
                investigation_id,

            "complaint_id":
                complaint_id,

            "status":
                status,

            "investigation_summary":
                result.get(
                    "investigation_summary"
                ),

            "root_cause_hypothesis":
                result.get(
                    "root_cause_hypothesis"
                ),

            "evidence_strength":
                result.get(
                    "evidence_strength"
                ),

            "confidence":
                result.get(
                    "confidence"
                ),

            "requires_human_review":
                human_review_required,

            "errors":
                result.get(
                    "errors",
                    [],
                ),
        }

    # --------------------------------------------------
    # Human review / resume
    # --------------------------------------------------

    def review_investigation(
        self,
        investigation_id: str,
        approved: bool,
        comment: str | None = None,
    ) -> Dict[str, Any]:

        investigation_id = (
            investigation_id
            .strip()
        )

        if not investigation_id:

            raise ValueError(
                "Investigation ID "
                "cannot be empty."
            )

        config = (
            self._build_config(
                investigation_id
            )
        )

        resume_payload = {
            "approved":
                approved,

            "comment":
                comment,
        }

        try:

            result = self.graph.invoke(
                Command(
                    resume=resume_payload
                ),
                config=config,
            )

        except Exception as error:

            raise RuntimeError(
                "Investigation resume failed: "
                f"{error}"
            ) from error

        complaint_id = result.get(
            "complaint_id",
            "",
        )

        human_approved = result.get(
            "human_approved",
            approved,
        )

        if human_approved:

            status = "completed"

        else:

            status = "rejected"

        return {
            "investigation_id":
                investigation_id,

            "complaint_id":
                complaint_id,

            "status":
                status,

            "human_approved":
                human_approved,

            "human_comment":
                result.get(
                    "human_comment",
                    comment,
                ),

            "final_report":
                result.get(
                    "final_report"
                ),

            "errors":
                result.get(
                    "errors",
                    [],
                ),
        }


# ------------------------------------------------------
# Shared application-level service instance
# ------------------------------------------------------

investigation_service = (
    InvestigationService()
)
