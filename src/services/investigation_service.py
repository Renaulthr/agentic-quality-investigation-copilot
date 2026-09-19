import time
import uuid
from typing import Any, Dict, Optional

import mlflow
from langgraph.types import Command

from src.agents.investigation_graph import build_investigation_graph
from src.logging_config.logger import get_logger
from src.tools.complaint_tool import get_complaint


logger = get_logger(__name__)


class InvestigationService:
    """
    Service layer for managing quality investigations.

    Responsibilities:
    - Validate complaint IDs.
    - Start LangGraph investigation workflows.
    - Preserve investigation/thread identity.
    - Handle human-in-the-loop interruptions.
    - Resume investigations after human review.
    - Produce structured operational logs.
    """

    def __init__(self) -> None:
        # Build once so the same graph/checkpointer is reused.
        self.graph = build_investigation_graph()

    @staticmethod
    def _build_config(
        investigation_id: str,
    ) -> Dict[str, Any]:
        """
        LangGraph uses investigation_id as thread_id.

        With the persistent SQLite checkpointer, this allows the
        investigation to be resumed even after an application restart.
        """
        return {
            "configurable": {
                "thread_id": investigation_id,
            }
        }

    @staticmethod
    def _extract_interrupt(
        result: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Extract the LangGraph interrupt payload when the workflow
        pauses for human review.
        """
        interrupts = result.get("__interrupt__")

        if not interrupts:
            return None

        interrupt = interrupts[0]

        return getattr(
            interrupt,
            "value",
            interrupt,
        )

    @staticmethod
    def _latency_ms(
        start_time: float,
    ) -> float:
        """Calculate elapsed execution time in milliseconds."""
        return round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

    @mlflow.trace(
        name="start_investigation",
        span_type="CHAIN",
    )
    def start_investigation(
        self,
        complaint_id: str,
    ) -> Dict[str, Any]:
        """
        Start a new quality investigation.

        The workflow executes until the LangGraph human-review
        interrupt is reached.
        """
        start_time = time.perf_counter()

        complaint_id = complaint_id.strip()

        # ---------------------------------------------------------
        # 1. Validate complaint
        # ---------------------------------------------------------

        complaint = get_complaint(complaint_id)

        if not complaint:
            logger.warning(
                "Investigation rejected because complaint was not found",
                extra={
                    "complaint_id": complaint_id,
                    "workflow_stage": "validation",
                    "status": "invalid_complaint",
                    "latency_ms": self._latency_ms(start_time),
                },
            )

            raise ValueError(
                f"Complaint '{complaint_id}' was not found."
            )

        # ---------------------------------------------------------
        # 2. Create investigation identity
        # ---------------------------------------------------------

        investigation_id = f"INV-{uuid.uuid4().hex[:12].upper()}"

        config = self._build_config(
            investigation_id
        )

        logger.info(
            "Investigation started",
            extra={
                "investigation_id": investigation_id,
                "complaint_id": complaint_id,
                "workflow_stage": "investigation_start",
                "status": "started",
            },
        )

        # ---------------------------------------------------------
        # 3. Execute LangGraph workflow
        # ---------------------------------------------------------

        try:
            result = self.graph.invoke(
                {
                    "complaint_id": complaint_id,
                },
                config=config,
            )

            interrupt_payload = self._extract_interrupt(
                result
            )

            # -----------------------------------------------------
            # 4. Human-review interrupt reached
            # -----------------------------------------------------

            if interrupt_payload is not None:
                latency_ms = self._latency_ms(
                    start_time
                )

                logger.info(
                    "Investigation awaiting human review",
                    extra={
                        "investigation_id": investigation_id,
                        "complaint_id": complaint_id,
                        "workflow_stage": "human_review",
                        "status": "awaiting_human_review",
                        "latency_ms": latency_ms,
                    },
                )

                return {
                    "investigation_id": investigation_id,
                    "complaint_id": complaint_id,
                    "status": "awaiting_human_review",
                    "investigation_summary": result.get(
                        "investigation_summary"
                    ),
                    "root_cause_hypothesis": result.get(
                        "root_cause_hypothesis"
                    ),
                    "evidence_strength": result.get(
                        "evidence_strength"
                    ),
                    "confidence": result.get(
                        "confidence"
                    ),
                    "requires_human_review": True,
                    "errors": result.get(
                        "errors",
                        [],
                    ),
                }

            # -----------------------------------------------------
            # 5. Workflow completed without interrupt
            # -----------------------------------------------------

            latency_ms = self._latency_ms(
                start_time
            )

            logger.info(
                "Investigation completed without human interrupt",
                extra={
                    "investigation_id": investigation_id,
                    "complaint_id": complaint_id,
                    "workflow_stage": "investigation_complete",
                    "status": "completed",
                    "latency_ms": latency_ms,
                },
            )

            return {
                "investigation_id": investigation_id,
                "complaint_id": complaint_id,
                "status": "completed",
                "investigation_summary": result.get(
                    "investigation_summary"
                ),
                "root_cause_hypothesis": result.get(
                    "root_cause_hypothesis"
                ),
                "evidence_strength": result.get(
                    "evidence_strength"
                ),
                "confidence": result.get(
                    "confidence"
                ),
                "requires_human_review": False,
                "errors": result.get(
                    "errors",
                    [],
                ),
            }

        except Exception as error:
            latency_ms = self._latency_ms(
                start_time
            )

            logger.exception(
                "Investigation workflow failed",
                extra={
                    "investigation_id": investigation_id,
                    "complaint_id": complaint_id,
                    "workflow_stage": "investigation",
                    "status": "failed",
                    "latency_ms": latency_ms,
                    "error": str(error),
                },
            )

            raise RuntimeError(
                f"Investigation workflow failed: {error}"
            ) from error

    @mlflow.trace(
        name="review_investigation",
        span_type="CHAIN",
    )
    def review_investigation(
        self,
        investigation_id: str,
        approved: bool,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resume an investigation that is waiting at the
        human-approval interrupt.
        """
        start_time = time.perf_counter()

        investigation_id = investigation_id.strip()

        config = self._build_config(
            investigation_id
        )

        review_status = (
            "approved"
            if approved
            else "rejected"
        )

        logger.info(
            "Human review received",
            extra={
                "investigation_id": investigation_id,
                "workflow_stage": "human_review",
                "status": review_status,
            },
        )

        try:
            # -----------------------------------------------------
            # 1. Resume same LangGraph thread
            # -----------------------------------------------------

            result = self.graph.invoke(
                Command(
                    resume={
                        "approved": approved,
                        "comment": comment,
                    }
                ),
                config=config,
            )

            # -----------------------------------------------------
            # 2. Determine final workflow status
            # -----------------------------------------------------

            final_status = (
                "completed"
                if approved
                else "rejected"
            )

            latency_ms = self._latency_ms(
                start_time
            )

            logger.info(
                "Investigation review completed",
                extra={
                    "investigation_id": investigation_id,
                    "workflow_stage": "human_review",
                    "status": final_status,
                    "latency_ms": latency_ms,
                },
            )

            # -----------------------------------------------------
            # 3. Return API-safe result
            # -----------------------------------------------------

            return {
                "investigation_id": investigation_id,
                "complaint_id": result.get(
                    "complaint_id"
                ),
                "status": final_status,
                "human_approved": approved,
                "human_comment": comment,
                "final_report": result.get(
                    "final_report"
                ),
                "errors": result.get(
                    "errors",
                    [],
                ),
            }

        except Exception as error:
            latency_ms = self._latency_ms(
                start_time
            )

            logger.exception(
                "Human review workflow failed",
                extra={
                    "investigation_id": investigation_id,
                    "workflow_stage": "human_review",
                    "status": "failed",
                    "latency_ms": latency_ms,
                    "error": str(error),
                },
            )

            raise RuntimeError(
                f"Investigation review failed: {error}"
            ) from error


# Application-level singleton.
# The graph/checkpointer is constructed once and reused by API requests.
investigation_service = InvestigationService()