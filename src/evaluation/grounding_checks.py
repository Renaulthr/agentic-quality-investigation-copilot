REFUSAL_PHRASES = [
    "insufficient evidence",
    "not available in the retrieved",
    "not supported by the retrieved",
    "cannot determine from the retrieved",
]


def is_refusal(
    answer: str,
) -> bool:

    answer_lower = (
        answer
        .lower()
        .strip()
    )

    return any(
        phrase in answer_lower
        for phrase in REFUSAL_PHRASES
    )


def contains_unsupported_claim(
    answer: str,
    should_answer: bool,
):

    refused = is_refusal(
        answer
    )

    if not should_answer:
        return not refused

    return False