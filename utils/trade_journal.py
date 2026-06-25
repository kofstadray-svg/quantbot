"""
utils/trade_journal.py — Logs every trade signal and its outcome.

Called in two places:
  1. When a BUY signal fires → record_signal()
  2. When a position closes  → record_outcome()

The journal is a SQLite file at logs/trade_journal.db.
autoresearch.py reads it to score each indicator's predictive value.
"""
from __future__ import annotations
import sqlite3
from datetime import date
from pathlib import Path
from loguru import logger

DB_PATH = Path(__file__).parent.parent / "logs" / "trade_journal.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        _init(con)
        return con
    except sqlite3.OperationalError as e:
        # Stale rollback journal or corrupt file — nuke and recreate
        logger.warning(f"trade_journal DB error ({e}), resetting database file.")
        for suffix in ("", "-journal", "-wal", "-shm"):
            p = Path(str(DB_PATH) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        _init(con)
        return con


def _init(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT    NOT NULL,
            signal_date      TEXT    NOT NULL,   -- ISO date
            strategy         TEXT    DEFAULT 'momentum',
            claude_score     INTEGER,
            -- raw indicator values at signal time
            rsi              REAL,
            macd_signal      REAL,
            golden_cross     INTEGER,            -- 0/1
            stoch_k          REAL,
            obv_trend        TEXT,               -- rising/falling/flat
            vol_rsi          REAL,
            bb_position      REAL,
            rel_volume       REAL,
            -- entry tracking (for return_pct calculation)
            entry_price      REAL,               -- avg fill price at entry
            alpaca_order_id  TEXT,               -- Alpaca order UUID for matching
            -- outcome (filled in later by position_monitor.py)
            exit_date        TEXT,
            return_pct       REAL,               -- fractional, e.g. 0.12 = +12%
            exit_reason      TEXT,               -- target/stop/max_hold/manual
            profitable       INTEGER             -- 1=yes 0=no NULL=open
        )
    """)
    # Migrate existing DBs
    existing = {row[1] for row in con.execute("PRAGMA table_info(signals)").fetchall()}
    if "entry_price" not in existing:
        con.execute("ALTER TABLE signals ADD COLUMN entry_price REAL")
    if "alpaca_order_id" not in existing:
        con.execute("ALTER TABLE signals ADD COLUMN alpaca_order_id TEXT")
    if "strategy" not in existing:
        con.execute("ALTER TABLE signals ADD COLUMN strategy TEXT DEFAULT 'momentum'")
    con.commit()


def record_signal(
    ticker: str,
    indicators: dict,
    claude_score: int,
    signal_date: date | None = None,
    entry_price: float | None = None,
    alpaca_order_id: str | None = None,
    strategy: str = "momentum",
) -> int:
    """
    Log a new BUY signal.  Returns the row id for use when recording outcome.

    indicators       — the same dict passed to Claude (from _compute_indicators).
    entry_price      — avg fill price at entry; required for position_monitor to
                       compute return_pct automatically.
    alpaca_order_id  — Alpaca order UUID; stored for future reconciliation.
    """
    sd = (signal_date or date.today()).isoformat()
    with _conn() as con:
        # Dedup: if there is already an OPEN signal (no outcome yet) for this
        # ticker, don't create another. The bot re-signals the same name on
        # consecutive scans / multiple scan types, which previously created
        # duplicate rows that all got resolved with the same exit -- inflating
        # the track record. One open signal per ticker until it's closed.
        dup = con.execute(
            "SELECT id FROM signals WHERE ticker=? AND profitable IS NULL LIMIT 1",
            (ticker,),
        ).fetchone()
        if dup is not None:
            logger.debug(
                f"Journal | {ticker}: open signal #{dup['id']} already exists "
                f"-- skipping duplicate record."
            )
            return int(dup["id"])

        cur = con.execute("""
            INSERT INTO signals
              (ticker, signal_date, strategy, claude_score,
               rsi, macd_signal, golden_cross, stoch_k,
               obv_trend, vol_rsi, bb_position, rel_volume,
               entry_price, alpaca_order_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ticker, sd, strategy, claude_score,
            indicators.get("rsi_14"),
            indicators.get("macd_signal"),
            1 if indicators.get("golden_cross") else 0,
            indicators.get("stoch_k"),
            indicators.get("obv_trend"),
            indicators.get("volume_rsi"),
            indicators.get("bb_position"),
            indicators.get("rel_volume"),
            entry_price,
            alpaca_order_id,
        ))
        row_id = cur.lastrowid
    logger.info(
        f"Journal | signal #{row_id} recorded: {ticker} score={claude_score}"
        + (f"  entry={entry_price:.2f}" if entry_price else "")
    )
    return row_id


def record_outcome(
    signal_id: int,
    return_pct: float,
    exit_reason: str,
    exit_date: date | None = None,
) -> None:
    """Fill in the outcome for a previously logged signal."""
    ed = (exit_date or date.today()).isoformat()
    profitable = 1 if return_pct > 0 else 0
    with _conn() as con:
        con.execute("""
            UPDATE signals
               SET exit_date=?, return_pct=?, exit_reason=?, profitable=?
             WHERE id=?
        """, (ed, return_pct, exit_reason, profitable, signal_id))
    logger.info(
        f"Journal | outcome #{signal_id}: {return_pct:+.2%} ({exit_reason})"
    )


def get_closed_trades(min_trades: int = 20) -> list[dict]:
    """Return all closed trades as dicts.  Returns [] if fewer than min_trades."""
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM signals WHERE profitable IS NOT NULL
            ORDER BY signal_date DESC
        """).fetchall()
    if len(rows) < min_trades:
        return []
    return [dict(r) for r in rows]


def performance_by_strategy() -> dict[str, dict]:
    """
    Return per-strategy performance stats for all closed trades.

    Example output:
        {
            "momentum":              {"trades": 45, "win_rate": 0.51, "avg_return": 0.009, "profit_factor": 1.4},
            "institutional_breakout":{"trades": 12, "win_rate": 0.67, "avg_return": 0.031, "profit_factor": 2.8},
        }
    Strategies with < 5 closed trades are included but marked with low-n warning.
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT strategy,
                   COUNT(*)                                         AS n,
                   SUM(CASE WHEN profitable=1 THEN 1 ELSE 0 END)   AS wins,
                   AVG(return_pct)                                  AS avg_ret,
                   SUM(CASE WHEN return_pct > 0 THEN return_pct ELSE 0 END) AS gross_win,
                   ABS(SUM(CASE WHEN return_pct < 0 THEN return_pct ELSE 0 END)) AS gross_loss
              FROM signals
             WHERE profitable IS NOT NULL
             GROUP BY strategy
             ORDER BY strategy
        """).fetchall()

    result: dict[str, dict] = {}
    for r in rows:
        strat = r["strategy"] or "momentum"
        n     = r["n"]
        wins  = r["wins"] or 0
        avg   = r["avg_ret"] or 0.0
        gw    = r["gross_win"] or 0.0
        gl    = r["gross_loss"] or 0.0
        pf    = round(gw / gl, 2) if gl > 0 else None
        result[strat] = {
            "trades":        n,
            "win_rate":      round(wins / n, 3) if n else 0.0,
            "avg_return":    round(avg, 4),
            "profit_factor": pf,
            "low_n":         n < 5,
        }
    return result


def win_rate_by_indicator() -> dict:
    """
    Returns a dict of indicator → win_rate for closed trades.
    Only indicators with ≥10 data points are included.

    Example output:
      {
        "golden_cross_true":  {"win_rate": 0.62, "n": 47},
        "obv_trend_rising":   {"win_rate": 0.58, "n": 52},
        "rsi_oversold":       {"win_rate": 0.71, "n": 14},
        ...
      }
    """
    trades = get_closed_trades(min_trades=0)
    if not trades:
        return {}

    buckets: dict[str, list[int]] = {}

    def _add(key: str, profitable: int) -> None:
        buckets.setdefault(key, []).append(profitable)

    for t in trades:
        p = t["profitable"]
        if p is None:
            continue

        # Golden cross
        if t["golden_cross"] is not None:
            _add("golden_cross_true" if t["golden_cross"] else "golden_cross_false", p)

        # OBV
        if t["obv_trend"]:
            _add(f"obv_{t['obv_trend']}", p)

        # RSI buckets
        rsi = t["rsi"]
        if rsi is not None:
            if rsi < 35:   _add("rsi_oversold",  p)
            elif rsi < 50: _add("rsi_healthy",   p)
            elif rsi < 70: _add("rsi_elevated",  p)
            else:           _add("rsi_overbought", p)

        # MACD
        macd = t["macd_signal"]
        if macd is not None:
            _add("macd_bullish" if macd > 0 else "macd_bearish", p)

        # Stochastic
        sk = t["stoch_k"]
        if sk is not None:
            if sk < 30:   _add("stoch_oversold",  p)
            elif sk > 70: _add("stoch_overbought", p)
            else:          _add("stoch_neutral",   p)

        # Volume RSI
        vr = t["vol_rsi"]
        if vr is not None:
            _add("vol_rsi_bull" if vr > 55 else "vol_rsi_bear", p)

        # Bollinger
        bb = t["bb_position"]
        if bb is not None:
            if bb < 0.2:   _add("bb_near_lower", p)
            elif bb > 0.8: _add("bb_near_upper", p)
            else:           _add("bb_middle",     p)

    result = {}
    for key, outcomes in buckets.items():
        if len(outcomes) >= 10:
            result[key] = {
                "win_rate": round(sum(outcomes) / len(outcomes), 3),
                "n": len(outcomes),
            }

    return result
