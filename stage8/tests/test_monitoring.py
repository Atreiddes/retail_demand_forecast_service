"""Тесты гейта деградации: чистая логика без БД."""
from forecast_service import monitoring as m


def test_gate_ok():
    acc = {"n_points": 100, "wmape": 0.3, "bias": 0.05, "coverage": 0.8}
    rep = m.gate(acc, [{"feature": "units", "psi": 0.03}])
    assert rep["ok"]
    assert rep["warnings"] == []


def test_gate_bias_and_coverage():
    acc = {"n_points": 100, "wmape": 0.3, "bias": 0.3, "coverage": 0.5}
    rep = m.gate(acc, None)
    assert not rep["ok"]
    assert any("смещение" in w for w in rep["warnings"])
    assert any("покрытие" in w for w in rep["warnings"])


def test_gate_wmape():
    acc = {"n_points": 100, "wmape": 0.9, "bias": 0.0, "coverage": 0.8}
    rep = m.gate(acc, None)
    assert not rep["ok"]
    assert any("WMAPE" in w for w in rep["warnings"])


def test_gate_drift():
    rep = m.gate(None, [{"feature": "sell_price", "psi": 0.4}])
    assert not rep["ok"]
    assert any("дрейф" in w for w in rep["warnings"])


def test_gate_no_data():
    rep = m.gate(None, None)
    assert rep["ok"]


def test_gate_planning_bias():
    bd = {"planning_bias": 0.2, "promo": {}, "fva_ma4": None}
    rep = m.gate(None, None, bd)
    assert not rep["ok"]
    assert any("плановое смещение" in w for w in rep["warnings"])


def test_gate_fva_collapsed():
    bd = {"planning_bias": 0.01, "promo": {}, "fva_ma4": {"improvement_pct": -3.0}}
    rep = m.gate(None, None, bd)
    assert not rep["ok"]
    assert any("MA-4" in w for w in rep["warnings"])


def test_gate_promo_poor():
    bd = {"planning_bias": 0.01, "promo": {"promo": {"wmape": 0.9, "bias": 0.0, "coverage": 0.8}},
          "fva_ma4": {"improvement_pct": 5.0}}
    rep = m.gate(None, None, bd)
    assert not rep["ok"]
    assert any("промо" in w for w in rep["warnings"])


def test_gate_breakdowns_ok():
    bd = {"planning_bias": 0.02, "promo": {"promo": {"wmape": 0.3, "bias": 0.0, "coverage": 0.8}},
          "fva_ma4": {"improvement_pct": 4.0}}
    rep = m.gate(None, None, bd)
    assert rep["ok"]


def test_gate_completeness_low():
    health = {"freshness": {"completeness": 0.7}, "churn": None, "revision_volatility": None}
    rep = m.gate(None, None, None, health)
    assert not rep["ok"]
    assert any("полнота" in w for w in rep["warnings"])


def test_gate_revision_high():
    health = {"freshness": None, "churn": None, "revision_volatility": 0.5}
    rep = m.gate(None, None, None, health)
    assert not rep["ok"]
    assert any("разброс" in w for w in rep["warnings"])


def test_gate_health_ok():
    health = {"freshness": {"completeness": 0.98}, "churn": {"new_series": 1, "dead_series": 0},
              "revision_volatility": 0.05}
    rep = m.gate(None, None, None, health)
    assert rep["ok"]
