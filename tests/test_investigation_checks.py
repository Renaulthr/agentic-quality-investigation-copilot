from src.evaluation.investigation_checks import (
    check_expected_value,
    check_source_present,
    contains_confirmed_root_cause,
    evaluate_rca_safety,
)


def test_expected_value():

    assert check_expected_value(
        "HIGH RISK",
        "HIGH RISK",
    )

    assert not check_expected_value(
        "LOW RISK",
        "HIGH RISK",
    )


def test_source_present():

    sources = [
        {
            "source":
                "pfmea_piston_assembly.md"
        },
        {
            "source":
                "historical_8d_cases.md"
        },
    ]

    assert check_source_present(
        sources,
        "historical_8d_cases.md",
    )


def test_confirmed_root_cause_detection():

    assert contains_confirmed_root_cause(
        "The root cause is excessive assembly force."
    )

    assert not contains_confirmed_root_cause(
        "The evidence suggests excessive assembly force "
        "as a possible cause."
    )


def test_rca_safety():

    result = evaluate_rca_safety(
        hypothesis=(
            "A possible cause is abnormal "
            "assembly force."
        ),
        must_not_confirm_root_cause=True,
    )

    assert result["passed"]