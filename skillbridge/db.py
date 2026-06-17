"""PostgreSQL access layer.

Provides:
- async connection pool for the API hot path
- sync context manager for pipeline batch work
- helpers for executing schema files and bulk operations
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from config import DB

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- Async pool
_async_pool: AsyncConnectionPool | None = None


async def init_pool(min_size: int = 2, max_size: int = 10) -> AsyncConnectionPool:
    global _async_pool
    if _async_pool is None:
        _async_pool = AsyncConnectionPool(
            conninfo=DB.dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        await _async_pool.open()
        log.info("Async DB pool opened (min=%d, max=%d)", min_size, max_size)
    return _async_pool


async def close_pool() -> None:
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None


@asynccontextmanager
async def acquire() -> AsyncIterator[psycopg.AsyncConnection]:
    if _async_pool is None:
        await init_pool()
    assert _async_pool is not None
    async with _async_pool.connection() as conn:
        # 3-second statement timeout protects pool from runaway queries.
        await conn.execute("SET LOCAL statement_timeout = '3s'")
        yield conn


async def fetch_one(sql: str, params: tuple | None = None) -> dict | None:
    async with acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params or ())
            return await cur.fetchone()


async def fetch_all(sql: str, params: tuple | None = None) -> list[dict]:
    async with acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params or ())
            return await cur.fetchall()


async def execute(sql: str, params: tuple | None = None) -> int:
    async with acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params or ())
            return cur.rowcount


# ----------------------------------------------------------------- Sync pool
# Used by the pipeline. Pipelines run as separate processes; they don't share
# the async pool with the API.
_sync_pool: ConnectionPool | None = None


def get_sync_pool() -> ConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        _sync_pool = ConnectionPool(
            conninfo=DB.dsn,
            min_size=1,
            max_size=4,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _sync_pool


@contextmanager
def sync_conn() -> Iterator[psycopg.Connection]:
    pool = get_sync_pool()
    with pool.connection() as conn:
        yield conn


@contextmanager
def sync_cursor() -> Iterator[psycopg.Cursor]:
    with sync_conn() as conn:
        with conn.cursor() as cur:
            yield cur


# -------------------------------------------------------- Utility operations
def executescript(sql_path: Path | str) -> None:
    path = Path(sql_path)
    sql_text = path.read_text(encoding="utf-8")
    log.info("Executing %s (%d bytes)", path.name, len(sql_text))
    with sync_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()


def upsert_returning(
    table: str,
    rows: list[dict[str, Any]],
    conflict_cols: list[str],
    update_cols: list[str] | None = None,
    returning: str = "*",
) -> list[dict]:
    """Generic upsert. Returns inserted/updated rows."""
    if not rows:
        return []
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    conflict = ", ".join(conflict_cols)
    if update_cols is None:
        update_cols = [c for c in cols if c not in conflict_cols]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause} "
        f"RETURNING {returning}"
    )
    out: list[dict] = []
    with sync_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, tuple(r[c] for c in cols))
                fetched = cur.fetchone()
                if fetched:
                    out.append(fetched)
        conn.commit()
    return out
