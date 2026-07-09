"""Гейт деградации модели: сводит точность прогноз-факт, калибровку интервалов и дрейф
входа в набор предупреждений и флаг ok. Пробой гейта - сигнал к переобучению (см. dags).

Логика чистая (без чтения БД и файлов): на вход готовые точность и дрейф, на выход отчёт.
Пороги калиброваны по свипу бэктестов на 10 магазинах FOODS M5 (origin 2016-03-19):
WMAPE p90=0.43, модуль смещения p90=0.12, покрытие min=0.66, FVA магазина min=-5.9%,
стабильность 0.06. Взяты за наблюдаемым разбросом с запасом; в рабочей системе пересчитываются.
"""
from __future__ import annotations

WMAPE_GATE = 0.5           # WMAPE выше нормального разброса (p90 0.43) - точность просела
BIAS_GATE = 0.2            # модуль смещения на листе выше наблюдаемого (max 0.13)
COVERAGE_FLOOR = 0.6       # покрытие ниже наблюдаемого минимума (0.66) - интервалы сузились
PSI_GATE = 0.25            # PSI признака выше - заметный дрейф входа
PLANNING_BIAS_GATE = 0.15  # плановое смещение выше наблюдаемого разброса (p90 0.12)
FVA_FLOOR = -8.0           # проигрыш MA-4 глубже шума одного магазина (min -5.9%) - реальный провал
COMPLETENESS_FLOOR = 0.9   # полнота последней недели факта ниже - данные загрузились не полностью
REVISION_GATE = 0.3        # разброс P50 по origin выше наблюдаемого (0.06) - прогноз дёргается


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
