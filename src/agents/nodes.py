from src.agents.state import InvestigationState

from src.tools.evidence_collector import (
    collect_investigation_evidence,
)
from src.rag.retriever import (
    retrieve_documents,
)

from langgraph.types import (
    interrupt,
)

import mlflow

from src.llm.ollama_llm import OllamaLLM

llm = OllamaLLM(
    model="llama3.2"
)

def human_approval_node(
    state: InvestigationState,
) -> InvestigationState:

    approval = interrupt(
        {
            "message":
                "Review the proposed RCA hypothesis.",

            "complaint_id":
                state.get(
                    "complaint_id"
                ),

            "hypothesis":
                state.get(
                    "root_cause_hypothesis"
                ),

            "evidence_strength":
                state.get(
                    "evidence_strength"
                ),

            "confidence":
                state.get(
                    "confidence"
                ),

            "instruction":
                "Approve or reject the hypothesis.",
        }
    )

    if isinstance(
        approval,
        dict,
    ):

        approved = bool(
            approval.get(
                "approved",
                False,
            )
        )

        comment = approval.get(
            "comment",
            "",
        )

    else:

        approved = bool(
            approval
        )

        comment = ""

    return {
        **state,

        "human_approved":
            approved,

        "human_comment":
            comment,
    }
@mlflow.trace(
    name="retrieve_quality_evidence",
    span_type="RETRIEVER",
)
def retrieve_quality_evidence_node(
    state: InvestigationState,
) -> dict:

    complaint = (
        state.get("complaint", {})
        .get("data", {})
    )

    issue = complaint.get(
        "Issue",
        "",
    )

    if not issue:
        return {
            "rag_evidence": [],
            "errors": (
                state.get("errors", [])
                + ["Complaint issue not available for RAG retrieval."]
            ),
        }

    results = retrieve_documents(
        query=issue,
        top_k=4,
    )

    rag_evidence = []

    for item in results:
        rag_evidence.append(
            {
                "source": item.get(
                    "source",
                    "",
                ),
                "chunk_id": item.get(
                    "chunk_id",
                    "",
                ),
                "distance": item.get(
                    "distance",
                ),
                "content": item.get(
                    "content",
                    "",
                ),
            }
        )

    return {
        "rag_evidence": rag_evidence,
    }

@mlflow.trace(
    name="synthesize_evidence",
    span_type="CHAIN",
)

def synthesize_evidence_node(
    state: InvestigationState,
) -> InvestigationState:

    complaint = state.get(
        "complaint",
        {}
    )

    traceability = state.get(
        "traceability",
        {}
    )

    quality_risk = state.get(
        "quality_risk",
        {}
    )

    historical_cases = state.get(
        "historical_cases",
        {}
    )

    rag_evidence = state.get(
        "rag_evidence",
        [],
    )

    prompt = f"""
You are a Manufacturing Quality Investigation Copilot.

Summarize the investigation evidence below.

STRICT RULES:
1. Use only the supplied evidence.
2. Do not declare root cause.
3. Separate confirmed facts from possible indications.
4. Mention ML risk output as model evidence, not as proof of root cause.
5. Do not invent missing information.
6. Keep the summary concise.

Complaint:
{complaint}

Traceability:
{traceability}

ML Quality Risk:
{quality_risk}

Historical Cases:
{historical_cases}

Retrieved Quality Documents:
{rag_evidence}

Return:

Confirmed Evidence:
- ...

Process Risk Indicators:
- ...

Historical Similarities:
- ...

Evidence Gaps:
- ...
"""

    summary = llm.generate(
        prompt
    )

    return {
        **state,
        "investigation_summary":
            summary,
    }

@mlflow.trace(
    name="load_investigation_evidence",
    span_type="CHAIN",
)
def load_evidence_node(
    state: InvestigationState,
) -> InvestigationState:

    complaint_id = state.get(
        "complaint_id"
    )

    if not complaint_id:
        return {
            **state,
            "errors": [
                "complaint_id is missing."
            ],
        }

    try:
        evidence = (
            collect_investigation_evidence(
                complaint_id
            )
        )

        return {
            **state,
            "complaint":
                evidence["complaint"],

            "traceability":
                evidence["traceability"],

            "quality_risk":
                evidence["quality_risk"],

            "historical_cases":
                evidence["historical_cases"],
        }

    except Exception as error:

        existing_errors = state.get(
            "errors",
            [],
        )

        return {
            **state,
            "errors":
                existing_errors
                + [str(error)],
        }

@mlflow.trace(
    name="generate_rca_hypothesis",
    span_type="CHAIN",
)

def generate_rca_hypothesis_node(
    state: InvestigationState,
) -> InvestigationState:

    summary = state.get(
        "investigation_summary",
        ""
    )

    if not summary:
        return {
            **state,
            "errors":
                state.get(
                    "errors",
                    []
                )
                + [
                    "Investigation summary missing."
                ],
        }

    prompt = f"""
You are a Manufacturing Quality Investigation Copilot.

Based ONLY on the investigation evidence below,
generate a possible root-cause hypothesis.

STRICT RULES:

1. Do not claim a root cause is confirmed.
2. Use terms such as:
   - possible cause
   - probable cause
   - hypothesis
   - evidence suggests
3. Separate process-related evidence from
   post-process or handling evidence.
4. The ML model prediction is supporting evidence,
   not proof of root cause.
5. If evidence is conflicting, explicitly say so.
6. If evidence is insufficient, say that the
   root cause cannot yet be confirmed.
7. Do not invent evidence.

Investigation Summary:
{summary}

Return exactly this structure:

Root Cause Hypothesis:
<concise hypothesis>

Supporting Evidence:
- ...
- ...

Contradicting Evidence:
- ...

Additional Verification Required:
- ...
"""

    hypothesis = llm.generate(
        prompt
    )

    return {
        **state,        "root_cause_hypothesis":
            hypothesis,
    }

@mlflow.trace(
    name="evaluate_evidence_strength",
    span_type="CHAIN",
)
def evaluate_evidence_strength_node(
    state: InvestigationState,
) -> InvestigationState:

    score = 0

    complaint = state.get(
        "complaint",
        {}
    )

    traceability = state.get(
        "traceability",
        {}
    )

    quality_risk = state.get(
        "quality_risk",
        {}
    )

    historical_cases = state.get(
        "historical_cases",
        {}
    )

    rag_evidence = state.get(
        "rag_evidence",
        []
    )

    # Complaint record available
    if complaint.get("found"):
        score += 1

    # Traceability available
    if traceability.get("found"):
        score += 1

    # ML API prediction available
    if quality_risk.get("success"):
        score += 1

    # Historical evidence available
    if historical_cases.get(
        "matches"
    ):
        score += 1

    # RAG evidence available
    if rag_evidence:
        score += 1

    if score >= 5:

        evidence_strength = "Strong"
        confidence = "High"

    elif score >= 3:

        evidence_strength = "Moderate"
        confidence = "Medium"

    else:

        evidence_strength = "Weak"
        confidence = "Low"

    requires_human_review = True

    return {
        **state,

        "evidence_strength":
            evidence_strength,

        "confidence":
            confidence,

        "requires_human_review":
            requires_human_review,
    }    

@mlflow.trace(
    name="generate_final_report",
    span_type="CHAIN",
)
def generate_final_report_node(
    state: InvestigationState,
) -> InvestigationState:

    approved = state.get(
        "human_approved",
        False,
    )

    if not approved:

        final_report = """
Investigation status: NOT APPROVED

The proposed root-cause hypothesis was not approved
during human review.

Further investigation is required before closure.
""".strip()

        return {
            **state,
            "final_report":
                final_report,
        }

    prompt = f"""
You are a Manufacturing Quality Investigation Copilot.

Prepare a concise quality investigation report.

Use ONLY the information provided.

Complaint:
{state.get("complaint")}

Traceability:
{state.get("traceability")}

ML Quality Risk:
{state.get("quality_risk")}

Investigation Summary:
{state.get("investigation_summary")}

RCA Hypothesis:
{state.get("root_cause_hypothesis")}

Evidence Strength:
{state.get("evidence_strength")}

Confidence:
{state.get("confidence")}

Human Review Comment:
{state.get("human_comment")}

IMPORTANT:
The human has approved the hypothesis for reporting,
but do not describe it as scientifically proven unless
the evidence explicitly proves it.

Return:

Complaint Summary:
...

Confirmed Evidence:
- ...

ML / Process Risk Assessment:
- ...

Root Cause Assessment:
...

Evidence Strength:
...

Recommended Verification / Action:
- ...

Human Review:
Approved
"""

    report = llm.generate(
        prompt
    )

    return {
        **state,
        "final_report":
            report,
    }