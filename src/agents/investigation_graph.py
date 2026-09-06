from langgraph.graph import (
    START,
    END,
    StateGraph,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from src.agents.state import (
    InvestigationState,
)

from src.agents.nodes import (
    load_evidence_node,
    retrieve_quality_evidence_node,
    synthesize_evidence_node,
    generate_rca_hypothesis_node,
    evaluate_evidence_strength_node,
    human_approval_node,
    generate_final_report_node,
)


def build_investigation_graph():

    builder = StateGraph(
        InvestigationState
    )

    builder.add_node(
        "load_evidence",
        load_evidence_node,
    )

    builder.add_node(
        "retrieve_quality_evidence",
        retrieve_quality_evidence_node,
    )

    builder.add_node(
        "synthesize_evidence",
        synthesize_evidence_node,
    )

    builder.add_node(
        "generate_rca_hypothesis",
        generate_rca_hypothesis_node,
    )

    builder.add_node(
        "evaluate_evidence_strength",
        evaluate_evidence_strength_node,
    )

    builder.add_node(
        "human_approval",
        human_approval_node,
    )

    builder.add_node(
        "generate_final_report",
        generate_final_report_node,
    )

    builder.add_edge(
        START,
        "load_evidence",
    )

    builder.add_edge(
        "load_evidence",
        "retrieve_quality_evidence",
    )

    builder.add_edge(
        "retrieve_quality_evidence",
        "synthesize_evidence",
    )

    builder.add_edge(
        "synthesize_evidence",
        "generate_rca_hypothesis",
    )

    builder.add_edge(
        "generate_rca_hypothesis",
        "evaluate_evidence_strength",
    )

    builder.add_edge(
        "evaluate_evidence_strength",
        "human_approval",
    )

    builder.add_edge(
        "human_approval",
        "generate_final_report",
    )

    builder.add_edge(
        "generate_final_report",
        END,
    )

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )