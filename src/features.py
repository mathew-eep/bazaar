from __future__ import annotations

import json
import sqlite3
from bisect import bisect_left
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


PERK_FLAGS = [
    "perk_mining_speed",
    "perk_mining_fortune",
    "perk_slayer_xp",
    "perk_slayer_cost",
    "perk_farming_fortune",
    "perk_farming_wisdom",
    "perk_mythological",
    "perk_fishing",
    "perk_xp_boost",
    "perk_coin_bonus",
    "perk_jerry",
    "is_minister_perk",
]

MAYOR_IDS = {
    "Finnegan": 0,
    "Cole": 1,
    "Aatrox": 2,
    "Diana": 3,
    "Diaz": 4,
    "Foxy": 5,
    "Jerry": 6,
    "Marina": 7,
    "Paul": 8,
    "Derpy": 9,
}

SB_YEAR_HOURS = 446.0
SKYBLOCK_EPOCH_UTC = datetime(2019, 6, 11, 17, 55, 0, tzinfo=timezone.utc)
SB_HOUR_SECONDS = 50.0
PRICE_COLS = ["sell", "buy", "sell_volume", "buy_volume", "max_buy", "min_sell"]

MARKET_EVENT_KEYS = [
    "spooky_festival",
    "new_year_celebration",
    "season_of_jerry",
    "traveling_zoo_early_summer",
    "traveling_zoo_early_winter",
    "mythological_ritual",
    "fishing_festival",
    "carnival",
    "stonk_exchange",
]

EVENT_LAGS = [1, 3, 6, 24]
DETERMINISTIC_EVENTS = ["spooky_festival", "new_year_celebration"]


def load_item_tags(items_csv: str | Path) -> list[str]:
    df = pd.read_csv(items_csv)
    if "item_tag" not in df.columns:
        raise ValueError("top_items.csv must contain an item_tag column")
    return [str(tag).upper() for tag in df["item_tag"].dropna().tolist()]


def load_price_history(db_path: str | Path, item_tags: list[str]) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(str(db_path))
    try:
        out: dict[str, pd.DataFrame] = {}
        for tag in item_tags:
            df = pd.read_sql_query(
                """
                SELECT timestamp, sell, buy, sell_volume, buy_volume, max_buy, min_sell
                FROM price_history
                WHERE item_tag = ?
                ORDER BY timestamp
                """,
                con,
                params=(tag,),
            )
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
            out[tag] = df
        return out
    finally:
        con.close()


def compute_norm_stats(
    db_path: str | Path,
    item_tags: list[str],
    train_frac: float = 0.8,
    out_path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    history = load_price_history(db_path, item_tags)
    stats: dict[str, dict[str, dict[str, float]]] = {}

    for tag, df in history.items():
        n_train = max(1, int(len(df) * train_frac))
        train_df = df.iloc[:n_train]

        per_item: dict[str, dict[str, float]] = {}
        for col in PRICE_COLS:
            vals = np.log1p(train_df[col].astype(float).clip(lower=0.0).to_numpy())
            mean = float(np.mean(vals)) if len(vals) else 0.0
            std = float(np.std(vals)) if len(vals) else 1.0
            per_item[col] = {"mean": mean, "std": max(std, 1e-8)}

        stats[tag] = per_item

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_flags(event_active: str | None) -> tuple[float, float, float]:
    evt = (event_active or "none").strip().lower()
    return (
        1.0 if evt == "spooky" else 0.0,
        1.0 if evt == "winter" else 0.0,
        1.0 if evt == "new_year" else 0.0,
    )


def _phase_from_sb_hour(sb_hour: float) -> tuple[float, float]:
    phase = 2.0 * np.pi * (sb_hour % SB_YEAR_HOURS) / SB_YEAR_HOURS
    return float(np.sin(phase)), float(np.cos(phase))


def _absolute_sb_hour(ts: pd.Timestamp) -> float:
    # Convert from UTC timestamp to SkyBlock hour index using fixed SkyBlock epoch.
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return float((dt - SKYBLOCK_EPOCH_UTC).total_seconds() / SB_HOUR_SECONDS)


def encode_game_state_row(row: pd.Series, active_perks: set[str]) -> np.ndarray:
    minister_present = bool(row.get("has_minister_perk", False))

    perk_vec = np.array(
        [
            1.0 if flag in active_perks else 0.0
            for flag in PERK_FLAGS[:-1]
        ]
        + [1.0 if minister_present else 0.0],
        dtype=np.float32,
    )

    sb_hour = _safe_float(row.get("sb_hour"), 0.0)
    sin_phase, cos_phase = _phase_from_sb_hour(sb_hour)

    mayor_name = str(row.get("mayor") or "")
    mayor_id = MAYOR_IDS.get(mayor_name, -1)

    cal = np.array(
        [
            sin_phase,
            cos_phase,
            _safe_float(row.get("sb_season"), 0.0) / 12.0,
            _safe_float(row.get("election_day"), 0.0) / 30.0,
            _safe_float(row.get("leading_pct"), 50.0) / 100.0,
            mayor_id / 10.0,
        ],
        dtype=np.float32,
    )

    ev = np.array(_event_flags(row.get("event_active")), dtype=np.float32)

    return np.concatenate([perk_vec, cal, ev]).astype(np.float32)


def load_game_state_features(db_path: str | Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        gs = pd.read_sql_query(
            """
            SELECT timestamp, mayor, minister, event_active, sb_season, sb_year,
                   election_day, leading_candidate, leading_pct
            FROM game_state
            ORDER BY timestamp
            """,
            con,
        )

        ap = pd.read_sql_query(
            """
            SELECT timestamp, perk_name, source
            FROM active_perks
            """,
            con,
        )

        try:
            ce = pd.read_sql_query(
                """
                SELECT timestamp, event_key
                FROM calendar_events
                """,
                con,
            )
        except Exception:
            ce = pd.DataFrame(columns=["timestamp", "event_key"])
    finally:
        con.close()

    if gs.empty:
        raise ValueError("game_state table is empty; cannot build known-future channel")

    gs["timestamp"] = pd.to_datetime(gs["timestamp"], utc=True, errors="coerce")
    gs = gs.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    ap["timestamp"] = pd.to_datetime(ap["timestamp"], utc=True, errors="coerce")
    ap = ap.dropna(subset=["timestamp"])

    perks_by_ts: dict[pd.Timestamp, set[str]] = {}
    minister_ts: set[pd.Timestamp] = set()
    for ts, perk, src in ap[["timestamp", "perk_name", "source"]].itertuples(index=False):
        perks_by_ts.setdefault(ts, set()).add(str(perk))
        if str(src) == "minister":
            minister_ts.add(ts)

    # Absolute SkyBlock hour index from fixed SkyBlock epoch.
    gs["sb_hour"] = gs["timestamp"].map(_absolute_sb_hour)
    gs["has_minister_perk"] = gs["timestamp"].isin(minister_ts)

    enc = []
    for row in gs.itertuples(index=False):
        ts = row.timestamp
        perks = perks_by_ts.get(ts, set())
        row_series = pd.Series(
            {
                "mayor": row.mayor,
                "event_active": row.event_active,
                "sb_season": row.sb_season,
                "election_day": row.election_day,
                "leading_pct": row.leading_pct,
                "sb_hour": row.sb_hour,
                "has_minister_perk": row.has_minister_perk,
            }
        )
        enc.append(encode_game_state_row(row_series, perks))

    feats = np.stack(enc, axis=0)
    feat_cols = [f"kf_{i}" for i in range(feats.shape[1])]

    ce["timestamp"] = pd.to_datetime(ce["timestamp"], utc=True, errors="coerce")
    ce = ce.dropna(subset=["timestamp", "event_key"])
    event_sets: dict[pd.Timestamp, set[str]] = {}
    for ts, key in ce[["timestamp", "event_key"]].itertuples(index=False):
        event_sets.setdefault(ts, set()).add(str(key))

    event_feat = np.zeros((len(gs), len(MARKET_EVENT_KEYS)), dtype=np.float32)
    for i, ts in enumerate(gs["timestamp"].tolist()):
        active = event_sets.get(ts, set())
        for j, key in enumerate(MARKET_EVENT_KEYS):
            event_feat[i, j] = 1.0 if key in active else 0.0

    # Deterministic event-position features (scheduled events only).
    # Feature pair per event: [hours_to_next_start_norm, hours_since_last_start_norm]
    ts_list = gs["timestamp"].tolist()
    ts_ns = np.array([int(ts.value) for ts in ts_list], dtype=np.int64)
    pos_feat = np.zeros((len(gs), len(DETERMINISTIC_EVENTS) * 2), dtype=np.float32)
    cap_hours = float(SB_YEAR_HOURS)

    for e_idx, event_key in enumerate(DETERMINISTIC_EVENTS):
        active_flags = np.array([event_key in event_sets.get(ts, set()) for ts in ts_list], dtype=bool)
        start_idx = np.where(active_flags & np.concatenate(([True], ~active_flags[:-1])))[0].tolist()
        if not start_idx:
            # Fallback to max distance when no event starts exist in loaded history.
            pos_feat[:, 2 * e_idx] = 1.0
            pos_feat[:, 2 * e_idx + 1] = 1.0
            continue

        for i in range(len(ts_list)):
            j = bisect_left(start_idx, i)
            next_i = start_idx[j] if j < len(start_idx) else None
            prev_i = start_idx[j - 1] if j > 0 else None

            if next_i is None:
                to_next_h = cap_hours
            else:
                delta_next = (ts_ns[next_i] - ts_ns[i]) / 3_600_000_000_000.0
                to_next_h = float(max(0.0, float(delta_next)))

            if prev_i is None:
                since_h = cap_hours
            else:
                delta_prev = (ts_ns[i] - ts_ns[prev_i]) / 3_600_000_000_000.0
                since_h = float(max(0.0, float(delta_prev)))

            pos_feat[i, 2 * e_idx] = np.float32(min(to_next_h / cap_hours, 1.0))
            pos_feat[i, 2 * e_idx + 1] = np.float32(min(since_h / cap_hours, 1.0))

    event_cols = [f"kf_evt_{k}" for k in MARKET_EVENT_KEYS]
    pos_cols = []
    for k in DETERMINISTIC_EVENTS:
        pos_cols.append(f"kf_evt_to_next_{k}")
        pos_cols.append(f"kf_evt_since_{k}")

    out = gs[["timestamp", "mayor", "minister"]].copy()
    out[feat_cols] = feats
    out[event_cols] = event_feat
    out[pos_cols] = pos_feat
    return out


def normalize_price_block(block: np.ndarray, stats: dict[str, dict[str, float]], cols: list[str]) -> np.ndarray:
    out = np.empty_like(block, dtype=np.float32)
    for i, col in enumerate(cols):
        vals = np.log1p(np.clip(block[:, i].astype(np.float64), a_min=0.0, a_max=None))
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        out[:, i] = ((vals - mean) / std).astype(np.float32)
    return out


def infer_sector_one_hot(item_tag: str) -> np.ndarray:
    t = item_tag.upper()

    mining_words = ("MITHRIL", "GEM", "REFINED", "TITANIUM", "UMBER", "JADE", "PERIDOT", "GREAT_WHITE_SHARK_TOOTH")
    farming_words = ("WHEAT", "CARROT", "POTATO", "CANE", "COMPOST", "CHOCO", "MELON", "PUMPKIN", "BURROWING_SPORES")
    combat_words = ("TARANTULA", "ESSENCE", "SHARD", "NULL", "HEMOBOMB", "SORROW", "SYNTHETIC_HEART")
    woods_words = ("OAK", "SPRUCE", "BIRCH", "DARK_OAK", "JUNGLE", "ACACIA")

    if any(w in t for w in mining_words):
        idx = 0
    elif any(w in t for w in farming_words):
        idx = 1
    elif any(w in t for w in combat_words):
        idx = 2
    elif any(w in t for w in woods_words):
        idx = 3
    else:
        idx = 4

    out = np.zeros(5, dtype=np.float32)
    out[idx] = 1.0
    return out


def infer_availability_one_hot(item_tag: str) -> np.ndarray:
    t = item_tag.upper()
    if any(w in t for w in ("CANDY", "SPOOKY", "JACK_O_LANTERN", "FALLEN_STAR")):
        idx = 1  # event-gated
    elif any(w in t for w in ("JERRY_BOX", "GIFT", "WINTER", "NEW_YEAR_CAKE")):
        idx = 2  # seasonal
    else:
        idx = 0  # free

    out = np.zeros(3, dtype=np.float32)
    out[idx] = 1.0
    return out


def build_static_context(
    item_tag: str,
    dependency_matrix: np.ndarray,
    item_index: dict[str, int],
    event_sensitivity: np.ndarray | None = None,
) -> np.ndarray:
    n = dependency_matrix.shape[0]
    row = np.zeros(n, dtype=np.float32)
    if item_tag in item_index:
        row = dependency_matrix[item_index[item_tag]].astype(np.float32, copy=True)

    sector = infer_sector_one_hot(item_tag)
    availability = infer_availability_one_hot(item_tag)
    parts = [row, sector, availability]
    if event_sensitivity is not None:
        parts.append(event_sensitivity.astype(np.float32, copy=False))
    return np.concatenate(parts).astype(np.float32)


def load_event_sensitivity_by_item(
    db_path: str | Path,
    item_tags: list[str],
) -> dict[str, np.ndarray]:
    """Build per-item static priors from event_price_correlation tables.

    Output vector layout:
      - For each MARKET_EVENT_KEYS: [return_lift, pearson_corr]
      - For each MARKET_EVENT_KEYS x EVENT_LAGS: return_lift_lag
    """
    base_dim = len(MARKET_EVENT_KEYS) * 2
    lag_dim = len(MARKET_EVENT_KEYS) * len(EVENT_LAGS)
    total_dim = base_dim + lag_dim

    out = {tag: np.zeros(total_dim, dtype=np.float32) for tag in item_tags}

    con = sqlite3.connect(str(db_path))
    try:
        corr_ok = bool(
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_price_correlation'"
            ).fetchone()
        )
        lag_ok = bool(
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_price_lag_correlation'"
            ).fetchone()
        )

        if corr_ok:
            corr = pd.read_sql_query(
                """
                SELECT event_key, item_tag, return_lift, pearson_corr
                FROM event_price_correlation
                """,
                con,
            )
            for row in corr.itertuples(index=False):
                key = str(row.event_key)
                tag = str(row.item_tag)
                if tag not in out or key not in MARKET_EVENT_KEYS:
                    continue
                k = MARKET_EVENT_KEYS.index(key)
                out[tag][2 * k] = float(row.return_lift if row.return_lift is not None else 0.0)
                out[tag][2 * k + 1] = float(row.pearson_corr if row.pearson_corr is not None else 0.0)

        if lag_ok:
            lag = pd.read_sql_query(
                """
                SELECT event_key, item_tag, lag_hours, return_lift
                FROM event_price_lag_correlation
                """,
                con,
            )
            for row in lag.itertuples(index=False):
                key = str(row.event_key)
                tag = str(row.item_tag)
                lag_h = int(row.lag_hours)
                if tag not in out or key not in MARKET_EVENT_KEYS or lag_h not in EVENT_LAGS:
                    continue
                k = MARKET_EVENT_KEYS.index(key)
                l = EVENT_LAGS.index(lag_h)
                idx = base_dim + k * len(EVENT_LAGS) + l
                out[tag][idx] = float(row.return_lift if row.return_lift is not None else 0.0)
    finally:
        con.close()

    return out
