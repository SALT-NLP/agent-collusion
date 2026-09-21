"""Read-only SQLite inspection and query tools."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from experiments.protocol.errors import (
    ToolError,
    tool_error_string,
)
from experiments.protocol.paths import _resolve_task_file_path
from experiments.protocol.state import _log_event, agent_state


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary {len(value)} bytes>"
    return value


def _sqlite_rows(
    cursor: sqlite3.Cursor, rows: list[sqlite3.Row]
) -> list[dict[str, Any]]:
    columns = [item[0] for item in cursor.description or []]
    result: list[dict[str, Any]] = []
    for row in rows:
        values = list(row)
        record: dict[str, Any] = {}
        for index, column in enumerate(columns):
            key = column
            suffix = 2
            while key in record:
                key = f"{column}__{suffix}"
                suffix += 1
            record[key] = _json_safe_value(values[index])
        result.append(record)
    return result


def _connect_readonly_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _resolve_database_path(
    state: dict[str, Any],
    slot: dict[str, Any],
    tool_name: str,
) -> Path:
    return _resolve_task_file_path(state, slot, "database_path", tool_name)


# Accepted statement keywords; mode=ro and query_only enforce read-only access.
READ_ONLY_STATEMENTS = frozenset({"select", "with"})


def _leading_keyword(sql: str) -> str:
    """Return the first SQL keyword, skipping leading comments and allowing adjacent symbols."""
    remaining = sql.lstrip()
    while True:
        if remaining.startswith("--"):
            _, newline, remaining = remaining.partition("\n")
            if not newline:
                return ""
        elif remaining.startswith("/*"):
            _, close, remaining = remaining.partition("*/")
            if not close:
                return ""
        else:
            break
        remaining = remaining.lstrip()
    match = re.match(r"[A-Za-z]+", remaining)
    return match.group(0).lower() if match else ""


# Expose small categorical domains so agents can use exact filter values.
MAX_DOMAIN_VALUES = 12

# One sample row shows date formats and numeric units.
SAMPLE_ROWS_PER_TABLE = 1


def _value_domains(
    conn: sqlite3.Connection, table_names: list[str]
) -> dict[str, dict[str, list[Any]]]:
    """Return bounded value domains for low-cardinality columns."""
    domains: dict[str, dict[str, list[Any]]] = {}
    for table in table_names:
        per_column: dict[str, list[Any]] = {}
        for column in conn.execute(f'PRAGMA table_info("{table}")'):
            name = str(column["name"])
            if "TEXT" not in str(column["type"]).upper():
                continue
            # Fetch one value beyond the cutoff to detect large domains cheaply.
            rows = conn.execute(
                f'SELECT DISTINCT "{name}" FROM "{table}" ORDER BY 1 LIMIT ?',
                (MAX_DOMAIN_VALUES + 1,),
            ).fetchall()
            # One distinct value filters nothing, so it is not a domain worth carrying.
            if 1 < len(rows) <= MAX_DOMAIN_VALUES:
                per_column[name] = [_json_safe_value(row[0]) for row in rows]
        if per_column:
            domains[table] = per_column
    return domains


def _inspect_database(state: dict[str, Any], actor: str) -> dict[str, Any]:
    slot = agent_state(state, actor)
    try:
        resolved = _resolve_database_path(state, slot, "inspect_database")
        with _connect_readonly_sqlite(resolved) as conn:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            table_names = [str(row["name"]) for row in table_rows]
            schema: dict[str, Any] = {}
            table_counts: dict[str, int] = {}
            sample_rows: dict[str, list[dict[str, Any]]] = {}
            foreign_keys: dict[str, list[dict[str, Any]]] = {}
            for table in table_names:
                columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                schema[table] = [
                    {
                        "name": str(column["name"]),
                        "type": str(column["type"]),
                        "notnull": bool(column["notnull"]),
                        "primary_key": bool(column["pk"]),
                    }
                    for column in columns
                ]
                count_row = conn.execute(
                    f'SELECT COUNT(*) AS n FROM "{table}"'
                ).fetchone()
                table_counts[table] = int(count_row["n"]) if count_row else 0
                fk_rows = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                foreign_keys[table] = [
                    {
                        "from": str(row["from"]),
                        "to_table": str(row["table"]),
                        "to": str(row["to"]),
                    }
                    for row in fk_rows
                ]
                cursor = conn.execute(
                    f'SELECT * FROM "{table}" LIMIT ?', (SAMPLE_ROWS_PER_TABLE,)
                )
                sample_rows[table] = _sqlite_rows(cursor, cursor.fetchall())
            value_domains = _value_domains(conn, table_names)
        result = {
            "success": True,
            "tables": table_names,
            "schema": schema,
            "foreign_keys": foreign_keys,
            "table_counts": table_counts,
            "value_domains": value_domains,
            "sample_rows": sample_rows,
        }
    except Exception as exc:
        # Database access failed before any agent-supplied query ran.
        result = {
            "success": False,
            "error": tool_error_string(
                exc,
                fallback_kind="TaskFileError",
                repo_root=state.get("repo_root", ""),
            ),
        }
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "inspect_database",
            "success": result["success"],
            "table_count": len(result.get("tables", [])),
            "domain_column_count": sum(
                len(columns) for columns in result.get("value_domains", {}).values()
            ),
            "schema_chars": len(
                json.dumps(result.get("schema", {}), ensure_ascii=False)
            ),
            **({"error": result["error"]} if not result["success"] else {}),
        },
    )
    return result


def _query_database(
    state: dict[str, Any],
    actor: str,
    sql: str,
    max_rows: int = 100,
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    try:
        resolved = _resolve_database_path(state, slot, "query_database")
        sql_text = sql.strip()
        if _leading_keyword(sql_text) not in READ_ONLY_STATEMENTS:
            raise ToolError(
                "PermissionError",
                "query_database only allows SELECT or WITH queries",
            )
        row_limit = max(1, min(int(max_rows or 100), 500))
        with _connect_readonly_sqlite(resolved) as conn:
            cursor = conn.execute(sql_text)
            fetched = cursor.fetchmany(row_limit + 1)
            truncated = len(fetched) > row_limit
            returned_rows = fetched[:row_limit]
            columns = [item[0] for item in cursor.description or []]
            rows = _sqlite_rows(cursor, returned_rows)
        result = {
            "success": True,
            "columns": columns,
            "truncated": truncated,
            "rows": rows,
        }
    except Exception as exc:
        result = {
            "success": False,
            "error": tool_error_string(
                exc,
                fallback_kind="QueryError",
                repo_root=state.get("repo_root", ""),
            ),
        }
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "query_database",
            "success": result["success"],
            "columns": result.get("columns", []),
            "truncated": result.get("truncated", False),
            **({"error": result["error"]} if not result["success"] else {}),
        },
    )
    return result
