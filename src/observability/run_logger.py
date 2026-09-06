import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

LOG_DIR = (
    PROJECT_ROOT
    / "reports"
    / "investigation_runs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_investigation_run(
    result: dict,
) -> Path:

    complaint_id = result.get(
        "complaint_id",
        "UNKNOWN",
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    file_path = (
        LOG_DIR
        / (
            f"{complaint_id}_"
            f"{timestamp}.json"
        )
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            default=str,
        )

    return file_path