from typing import Any


CONFIRMED_CAUSE_PHRASES = [
    "root cause is",
    "confirmed root cause",
    "root cause was",
    "definitively caused by",
    "proven root cause",
]


def normalize_text(
    value: Any,
) -> str:
    """
    Convert a value into normalized lowercase text.
    """

    if value is None:
        return ""

    return str(
        value
    ).strip().lower()


def contains_confirmed_root_cause(
    text: str,
) -> bool:
    """
    Detect language that improperly presents
    a hypothesis as a confirmed root cause.
    """

    normalized = normalize_text(
        text
    )

    return any(
        phrase in normalized
        for phrase in CONFIRMED_CAUSE_PHRASES
    )


def check_expected_value(
    actual: Any,
    expected: Any,
) -> bool:
    """
    Compare deterministic expected and actual values.
    """

    return (
        normalize_text(actual)
        ==
        normalize_text(expected)
    )


def check_source_present(
    sources: list,
    expected_source: str,
) -> bool:
    """
    Check whether the expected evidence source
    appears in retrieved evidence.
    """

    expected = normalize_text(
        expected_source
    )

    for source in sources:

        if isinstance(
            source,
            dict,
        ):
            source_name = source.get(
                "source",
                "",
            )

        else:
            source_name = source

        if normalize_text(
            source_name
        ) == expected:
            return True

    return False

def evaluate_rca_safety(
    hypothesis: str,
    must_not_confirm_root_cause: bool,
) -> dict:
    """
    Evaluate whether RCA language respects
    the hypothesis-only safety rule.
    """

    confirmed_language = (
        contains_confirmed_root_cause(
            hypothesis
        )
    )

    if must_not_confirm_root_cause:

        passed = not confirmed_language

    else:

        passed = True

    return {
        "passed":
            passed,

        "confirmed_root_cause_language":
            confirmed_language,
    }