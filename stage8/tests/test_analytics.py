"""Тесты аналитики мониторинга (crud) на синтетических данных.

Гоняются в отдельной временной базе (монки-патч движка), чтобы не пересекаться с основной.
"""
import datetime as dt
import os

import pytest
from sqlalchemy import create_engine, text
from sqlmodel import Session

from forecast_service import crud, db, models

BASE = dt.date(2015, 1, 5)  # понедельник
N_WEEKS = 50
SERIES = [
    ("FOODS_1_A", "DEPT_1", "CA", "CA_1", 5.0),   # частый
    ("FOODS_1_B", "DEPT_1", "CA", "CA_1", 0.2),   # прерывистый
    ("FOODS_2_A", "DEPT_2", "TX", "TX_1", 4.0),   # частый
    ("FOODS_2_B", "DEPT_2", "TX", "TX_1", 6.0),   # частый
]


def _week(i):
    return BASE + dt.timedelta(weeks=i)


def _seed_run(s, origin_i, units_of):
    run = models.ForecastRun(origin=_week(origin_i), horizon_weeks=8, status=models.COMPLETED,
                             n_series=len(SERIES), model_version="test")
    s.add(run)
    s.flush()
    for item, _dept, _state, store, _mu in SERIES:
        sid = f"{item}_{store}"
        for h in range(1, 9):
            p50 = max(units_of(item, origin_i + h), 0.0)
            s.add(models.ForecastPoint(run_id=run.id, series_id=sid, week_start_date=_week(origin_i + h),
                                       h=h, p10=p50 * 0.5, p50=p50, p90=p50 * 1.5 + 1))
    return run.id


def _urls():
    base = os.environ.get("DB_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/forecast")
    root = base.rpartition("/")[0]
    return f"{root}/postgres", f"{root}/forecast_analytics"


@pytest.fixture
def seeded(monkeypatch):
    admin_url, test_url = _urls()
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text("DROP DATABASE IF EXISTS forecast_analytics"))
        c.execute(text("CREATE DATABASE forecast_analytics"))
    eng = create_engine(test_url)
    monkeypatch.setattr(crud, "engine", eng)
    monkeypatch.setattr(db, "engine", eng)
    models.SQLModel.metadata.create_all(eng)

    with Session(eng) as s:
        for item, dept, state, store, _mu in SERIES:
            s.add(models.Series(id=f"{item}_{store}", item_id=item, dept_id=dept, cat_id="FOODS",
                                store_id=store, state_id=state))
        s.flush()
        units = {}
        for item, dept, state, store, mu in SERIES:
            sid = f"{item}_{store}"
            for i in range(N_WEEKS):
                u = round(mu) if mu >= 1 else (1 if i % 5 == 0 else 0)  # редкий: единицы раз в 5 недель
                units[(item, i)] = float(u)
                s.add(models.SalesHistory(series_id=sid, week_start_date=_week(i), units=u,
                                          revenue=u * 2.0, sell_price=2.0, snap_days=1,
                                          event_days=1 if i % 4 == 0 else 0, available_days=7, n_days=7))
        s.flush()
        r1 = _seed_run(s, 40, lambda it, i: units.get((it, i), 0.0))
        r2 = _seed_run(s, 38, lambda it, i: units.get((it, i), 0.0))  # пересечение недель с r1
        s.commit()

    yield r1, r2

    eng.dispose()
    with admin.connect() as c:
        c.execute(text("DROP DATABASE IF EXISTS forecast_analytics"))
    admin.dispose()


def test_accuracy_vs_actual(seeded):
    r1, _ = seeded
    acc = crud.accuracy_vs_actual(r1)
    assert acc is not None
    assert acc["n_points"] > 0
    assert acc["wmape"] >= 0
    assert 0.0 <= acc["coverage"] <= 1.0


def test_run_coverage(seeded):
    r1, _ = seeded
    assert crud.run_coverage(r1) == 1.0


def test_accuracy_breakdowns(seeded):
    r1, _ = seeded
    bd = crud.accuracy_breakdowns(r1)
    assert bd is not None
    assert set(bd["by_horizon"]) == {str(h) for h in range(1, 9)}
    assert set(bd["by_state"]) == {"CA", "TX"}
    assert "frequent" in bd["segments"] and "intermittent" in bd["segments"]
    assert bd["fva_ma4"]["improvement_pct"] is not None


def test_data_freshness(seeded):
    fr = crud.data_freshness()
    assert fr is not None
    assert fr["history_weeks"] == N_WEEKS
    assert 0.0 <= fr["completeness"] <= 1.0


def test_assortment_churn(seeded):
    ch = crud.assortment_churn()
    assert ch is not None
    assert ch["new_series"] >= 0 and ch["dead_series"] >= 0


def test_revision_volatility(seeded):
    vol = crud.revision_volatility()
    assert vol is None or vol >= 0.0  # два прогона с пересечением недель


def test_last_matured_run(seeded):
    r1, r2 = seeded
    assert crud.last_matured_run_id() == max(r1, r2)  # последний по id прогон с вызревшим фактом
