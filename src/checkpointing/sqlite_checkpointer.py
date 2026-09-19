import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_CHECKPOINT_PATH = "data/checkpoints/investigations.db"

CHECKPOINT_DB_PATH = os.getenv(
    "CHECKPOINT_DB_PATH",
    DEFAULT_CHECKPOINT_PATH,
)


def create_sqlite_checkpointer() -> SqliteSaver:
    """
    Create a persistent SQLite-backed LangGraph checkpointer.

    check_same_thread=False allows the connection to be used
    by FastAPI worker threads within the same process.
    """

    db_path = Path(CHECKPOINT_DB_PATH)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )

    return SqliteSaver(connection)