"""Гейт деградации модели: сводит точность прогноз-факт, калибровку интервалов и дрейф
входа в набор предупреждений и флаг ok. Пробой гейта - сигнал к переобучению (см. dags).

Логика чистая (без чтения БД и файлов): на вход готовые точность и дрейф, на выход отчёт.
Пороги - демо-значения для FOODS M5, в рабочей системе подбираются по своей истории.
"""
from __future__ import annotations

WMAPE_GATE = 0.6           # WMAPE свежего прогноза хуже - точность просела
BIAS_GATE = 0.15           # модуль смещения на листе больше - систематический пере- или недопрогноз
COVERAGE_FLOOR = 0.65      # доля факта в интервале P10-P90 ниже номинальных 0.8 с запасом - интервалы узкие
PSI_GATE = 0.25            # PSI признака выше - заметный дрейф входа
PLANNING_BIAS_GATE = 0.10  # смещение на плановом уровне (штат) больше - заказ поедет систематически
FVA_FLOOR = 0.0            # прирост над базой MA-4 не выше нуля - модель перестала бить наивную базу
COMPLETENESS_FLOOR = 0.9   # полнота последней недели факта ниже - данные загрузились не полностью
REVISION_GATE = 0.3        # разброс P50 по origin выше - прогноз слишком дёргается между прогонами


def gate(accuracy: dict | None, drift: list | None, breakdowns: dict | None = None,
         health: dict | None = None) -> dict:
    """Собирает сигналы деградации в предупреждения. accuracy - результат прогноз-факт
    (или None, если факта ещё нет), drift - признаки с PSI, breakdowns - разрезы точности,
    health - свежесть данных, дрейф ассортимента и стабильность прогноза."""
    warnings = []
    if accuracy:
        wmape = accuracy.get("wmape")
        bias = accuracy.get("bias")
        coverage = accuracy.get("coverage")
        if wmape is not None and wmape > WMAPE_GATE:
            warnings.append(f"WMAPE {wmape:.2f} хуже порога {WMAPE_GATE}")
        if bias is not None and abs(bias) > BIAS_GATE:
            warnings.append(f"смещение {bias:+.2f} по модулю больше порога {BIAS_GATE}")
        if coverage is not None and coverage < COVERAGE_FLOOR:
            warnings.append(f"покрытие интервала {coverage:.2f} ниже порога {COVERAGE_FLOOR}")
    if drift:
        wide = [f["feature"] for f in drift if f.get("psi") is not None and f["psi"] > PSI_GATE]
        if wide:
            warnings.append(f"дрейф признаков: {', '.join(wide)} (PSI выше {PSI_GATE})")
    if breakdowns:
        pb = breakdowns.get("planning_bias")
        if pb is not None and pb > PLANNING_BIAS_GATE:
            warnings.append(f"плановое смещение {pb:.2f} больше порога {PLANNING_BIAS_GATE}")
        promo = (breakdowns.get("promo") or {}).get("promo")
        if promo and promo["wmape"] > WMAPE_GATE:
            warnings.append(f"WMAPE в промо-недели {promo['wmape']:.2f} хуже порога {WMAPE_GATE}")
        fva = breakdowns.get("fva_ma4")
        if fva and fva.get("improvement_pct") is not None and fva["improvement_pct"] <= FVA_FLOOR:
            warnings.append(f"модель не бьёт базу MA-4 (прирост {fva['improvement_pct']}%)")
    if health:
        freshness = health.get("freshness")
        if freshness and freshness.get("completeness") is not None \
                and freshness["completeness"] < COMPLETENESS_FLOOR:
            warnings.append(f"полнота последней недели {freshness['completeness']:.2f} ниже "
                            f"порога {COMPLETENESS_FLOOR}")
        rev = health.get("revision_volatility")
        if rev is not None and rev > REVISION_GATE:
            warnings.append(f"разброс прогноза по origin {rev:.2f} выше порога {REVISION_GATE}")
    return {"ok": not warnings, "warnings": warnings, "accuracy": accuracy,
            "drift": drift, "breakdowns": breakdowns, "health": health}
