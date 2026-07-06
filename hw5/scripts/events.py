"""Недельные признаки событий из календаря M5.

Тип события несёт сигнал: религиозные и национальные праздники двигают спрос иначе, чем
спортивные. Поднимаем из календаря четыре типа как недельные счётчики дней и флаги топ-событий.
Календарь известен наперёд, поэтому признаки целевой недели берём без утечки.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CAL = Path(__file__).resolve().parent.parent.parent / "hw4" / "data" / "processed" / "calendar.parquet"

TYPES = ["Cultural", "National", "Religious", "Sporting"]
TYPE_COLS = [f"ev_{t.lower()}" for t in TYPES]
# Топ-события с заметным спросовым следом, флаг на неделю.
NAMED = {
    "ev_christmas": "Christmas", "ev_thanksgiving": "Thanksgiving",
    "ev_superbowl": "SuperBowl", "ev_easter": "Easter",
    "ev_mothersday": "Mother's day", "ev_laborday": "LaborDay",
}
EVENT_FEATURES = TYPE_COLS + list(NAMED)


def weekly_event_table(calendar_path=CAL):
    """Таблица событий на wm_yr_wk: счётчики дней по 4 типам и флаги топ-событий."""
    c = pd.read_parquet(calendar_path)
    long = pd.concat([
        c[["wm_yr_wk", "event_name_1", "event_type_1"]].rename(
            columns={"event_name_1": "name", "event_type_1": "type"}),
        c[["wm_yr_wk", "event_name_2", "event_type_2"]].rename(
            columns={"event_name_2": "name", "event_type_2": "type"}),
    ], ignore_index=True).dropna(subset=["name"])

    rows = c[["wm_yr_wk"]].drop_duplicates().set_index("wm_yr_wk")
    for t, col in zip(TYPES, TYPE_COLS):
        cnt = long[long["type"] == t].groupby("wm_yr_wk").size()
        rows[col] = cnt.reindex(rows.index).fillna(0).astype("int8")
    for col, name in NAMED.items():
        wks = long[long["name"] == name]["wm_yr_wk"].unique()
        rows[col] = rows.index.isin(wks).astype("int8")
    return rows.reset_index()


def add_event_features(full, calendar_path=CAL):
    """Добавить недельные признаки событий в панель по wm_yr_wk (целевая неделя, без утечки)."""
    tbl = weekly_event_table(calendar_path)
    out = full.merge(tbl, on="wm_yr_wk", how="left", validate="many_to_one")
    for col in EVENT_FEATURES:
        out[col] = out[col].fillna(0).astype("int8")
    return out
