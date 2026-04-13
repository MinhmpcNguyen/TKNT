from __future__ import annotations

import logging
from threading import Lock

from db.postgres import create_connection
from db.schema import (
    get_runtime_schema_statements,
    get_runtime_user_content_schema_statements,
)

logger = logging.getLogger(__name__)
_runtime_schema_lock = Lock()
_user_content_schema_lock = Lock()
_runtime_schema_initialized = False
_user_content_schema_initialized = False


def initialize_runtime_schema() -> None:
    _execute_schema_statements(
        get_runtime_schema_statements(),
        label="Runtime schema",
    )


def initialize_user_content_schema() -> None:
    _execute_schema_statements(
        get_runtime_user_content_schema_statements(),
        label="User content schema",
    )


def ensure_runtime_schema() -> None:
    global _runtime_schema_initialized
    if _runtime_schema_initialized:
        return
    with _runtime_schema_lock:
        if _runtime_schema_initialized:
            return
        initialize_runtime_schema()
        _runtime_schema_initialized = True


def ensure_user_content_schema() -> None:
    global _user_content_schema_initialized
    if _user_content_schema_initialized:
        return
    with _user_content_schema_lock:
        if _user_content_schema_initialized:
            return
        initialize_user_content_schema()
        _assert_tables_exist(
            [
                "user_accounts",
                "auth_sessions",
                "saved_layouts",
                "generated_renders",
            ]
        )
        _user_content_schema_initialized = True


def _execute_schema_statements(statements: list[str], *, label: str) -> None:
    failures: list[str] = []
    with create_connection() as conn:
        with conn.cursor() as cur:
            for index, statement in enumerate(statements, start=1):
                try:
                    cur.execute(statement)
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    failures.append(
                        f"{label} statement {index} failed: {exc} | sql={_statement_preview(statement)}"
                    )
    if failures:
        for failure in failures:
            logger.warning("%s", failure)
        logger.warning(
            "%s initialization completed with %d failed statements.",
            label,
            len(failures),
        )
        return
    logger.info("%s initialization completed with %d statements.", label, len(statements))


def _assert_tables_exist(table_names: list[str]) -> None:
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = ANY(%(table_names)s)
    """
    with create_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"table_names": table_names})
            rows = cur.fetchall()
    existing = {str(row["table_name"]) for row in rows}
    missing = [name for name in table_names if name not in existing]
    if missing:
        missing_tables = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing required runtime tables: {missing_tables}")


def _statement_preview(statement: str) -> str:
    condensed = " ".join(statement.split())
    if len(condensed) <= 140:
        return condensed
    return f"{condensed[:137]}..."
