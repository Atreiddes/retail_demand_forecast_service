"""Коррекция упущенных продаж (censored demand) на недельном уровне.

В недельной панели M5 available_days бинарный {0, 7}: недели вне ассортимента уже
исключены масками, а подозрительные нули (ряд обычно продаёт и доступен, но неделя
нулевая) лишь помечаются флагом - точно отделить сток-аут от истинного нуля без данных
о запасах нельзя.
"""
from __future__ import annotations

import pandas as pd

TCOL = "week_start_date"


def availability_summary(full):
    """Доли недель по доступности (подтверждает бинарность available_days на недельном уровне)."""
    full = full[full["n_days"] == 7]
    vc = full["available_days"].value_counts(normalize=True).sort_index()
    return {
        "binary_{0,7}": bool(set(full["available_days"].unique()) <= {0, 7}),
        "share_out_of_assortment": round(float(vc.get(0, 0.0)), 4),
        "share_available": round(float(vc.get(7, 0.0)), 4),
    }


def flag_suspect_zeros(full, min_nonzero_rate=0.5):
    """Пометить подозрительные нули: units==0 при доступности у ряда, который обычно продаёт."""
    f = full.copy()
    nz = f[f["available_days"] > 0].assign(_nz=(f["units"] > 0).astype(int))
    rate = nz.groupby("id", observed=True)["_nz"].mean()
    f["_rate"] = f["id"].map(rate).fillna(0.0)
    f["suspect_zero"] = (
        (f["units"] == 0) & (f["available_days"] == 7) & (f["_rate"] >= min_nonzero_rate)
    ).astype("int8")
    return f.drop(columns="_rate")


def report(full):
    """Сводка по упущенным продажам для evaluation/EDA."""
    s = availability_summary(full)
    flagged = flag_suspect_zeros(full)
    s["suspect_zero_weeks"] = int(flagged["suspect_zero"].sum())
    s["suspect_zero_share"] = round(float(flagged["suspect_zero"].mean()), 4)
    return s
