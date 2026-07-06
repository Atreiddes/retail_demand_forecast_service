"""Метрик-лог: ведётся прогонами, путь от простого бейзлайна до основной модели.

snapshot()/log_run() дописывают метрики моделей в metriclog.csv, render() строит metriclog.md
(сравнение вариантов, лог экспериментов, авто-лог основной модели). Совокупная метрика WRMSSE,
8 фолдов walk-forward, ниже лучше.
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG = ROOT.parent / "metriclog.csv"
MD = ROOT.parent / "metriclog.md"
MAIN = "lgb_ensemble"   # основная модель
REF = "moving_avg"      # простой референс

# вариант -> (роль, примечание). Роль задаёт порядок и группировку в сравнении.
VARIANTS = {
    "lgb_ensemble": ("основная", "среднее default и tuned LightGBM; над референсом MA-4"),
    "lgb_tweedie": ("компонент", "Tweedie + direct multi-horizon + калибровка смещения"),
    "lgb_optuna": ("компонент", "тюнинг гиперпараметров по WRMSSE"),
    "moving_avg": ("референс", "сильный простой: свежий уровень ряда, почти несмещён"),
    "bayes_structural": ("альтернатива", "другое семейство (numpyro), не оптимизирован под WRMSSE"),
    "seasonal_ma": ("пол", "уровень × сезонный индекс ряда; per-item сезонность шумная"),
    "croston_sba": ("пол", "прерывистые; плоская интенсивность занижает уровень на weekly"),
    "seasonal_naive": ("пол", "факт год назад; ассортимент меняется, год назад рядов не было"),
}
ORDER = {"основная": 0, "компонент": 1, "референс": 2, "альтернатива": 3, "пол": 4}

# Лог экспериментов: что пробовали -> результат -> вердикт (по фактическим прогонам).
EXPERIMENTS = [
    ("direct multi-horizon признаки", "лаги >= горизонта давали WRMSSE ~2.0 (хуже наива) -> признаки на момент origin -> 0.75", "взято, ключевое"),
    ("калибровка смещения (magic multiplier)", "Tweedie +12% bias, WRMSSE ~1.7 -> фактор на cal-блоке -> Bias ~0, WRMSSE 0.75", "взято, ключевое"),
    ("тюнинг Optuna по WRMSSE", "lgb_optuna 0.751 против дефолта 0.749", "прироста нет, ушло в ансамбль"),
    ("event one-hot (типы событий календаря)", "0.6842 -> 0.6728 на 3 фолдах", "помогает +1.7%, по умолчанию off для сравнимости"),
    ("middle-out reconciliation к MA-агрегату", "bottom-up 0.737 -> middle-out 0.906", "проигрыш, оставлен bottom-up"),
    ("MA × сезонный индекс (seasonal_ma)", "1.340 против MA-4 0.802", "хуже, per-item индекс шумный, не берём"),
    ("Croston/SBA (прерывистые)", "1.587 против MA-4 0.802", "хуже на weekly WRMSSE, не берём как основной"),
    ("квантили P10/P50/P90 (pinball)", "P90 покрытие 0.93, P10 завышен (0.28)", "есть, нижний квантиль требует калибровки"),
]
COLS = ["ts", "commit", "model", "WRMSSE", "RMSSE_L12", "MASE_L12", "Bias", "folds", "note"]


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _git_sha():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or "?"
    except Exception:
        return "?"


def log_run(model, metrics, folds=None, note="", ts=None, commit=None):
    """Дописать строку в metriclog.csv. metrics: dict с WRMSSE/RMSSE_L12/MASE_L12/Bias."""
    ts = ts or _now()
    commit = commit or _git_sha()
    new = not LOG.exists()
    with LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        w.writerow([ts, commit, model, metrics.get("WRMSSE"), metrics.get("RMSSE_L12"),
                    metrics.get("MASE_L12"), metrics.get("Bias"), folds, note])


def log_summary(summary_df, folds=None, note="", ts=None, commit=None):
    """Залогировать все модели из сводки (index=model; столбцы WRMSSE_mean.. или WRMSSE..)."""
    ts, commit = ts or _now(), commit or _git_sha()
    for m, r in summary_df.iterrows():
        log_run(m, {"WRMSSE": r.get("WRMSSE_mean", r.get("WRMSSE")),
                    "RMSSE_L12": r.get("RMSSE_L12"), "MASE_L12": r.get("MASE_L12"),
                    "Bias": r.get("Bias")}, folds=folds, note=note, ts=ts, commit=commit)


def _fmt(x, d=4):
    return f"{float(x):.{d}f}" if pd.notna(x) else "-"


def render():
    """Перестроить metriclog.md: сравнение вариантов, лог экспериментов, авто-лог основной модели."""
    if not LOG.exists():
        return
    log = pd.read_csv(LOG).dropna(subset=["WRMSSE"])
    latest = log.groupby("model").tail(1).set_index("model")
    rows = [(m, VARIANTS.get(m, ("альтернатива", ""))) for m in latest.index]
    rows.sort(key=lambda t: (ORDER.get(t[1][0], 9), float(latest.loc[t[0], "WRMSSE"])))

    out = [
        "# Метрик-лог",
        "",
        "Совокупная метрика WRMSSE (взвешенный RMSSE по 12 уровням иерархии M5), 8 фолдов",
        "walk-forward (обучение <= origin), ниже лучше. Ведётся автоматически прогонами",
        "(`run_cv.py`, `baselines_extra.py`); правится не руками, а кодом.",
        "",
        "## Сравнение вариантов",
        "",
        "| вариант | роль | WRMSSE | RMSSE_L12 | примечание |",
        "|---|---|---|---|---|",
    ]
    for m, (role, note) in rows:
        r = latest.loc[m]
        out.append(f"| {m} | {role} | {_fmt(r['WRMSSE'])} | {_fmt(r['RMSSE_L12'])} | {note} |")

    if MAIN in latest.index and REF in latest.index:
        main_w, ref_w = float(latest.loc[MAIN, "WRMSSE"]), float(latest.loc[REF, "WRMSSE"])
        impr = (ref_w - main_w) / ref_w * 100
        out += ["",
                f"Основная модель зажата: сильнейший простой референс MA-4 ({ref_w:.3f}) сверху",
                f"простых, полы (сезонный индекс, Croston, наив) снизу. Из простых per-series MA-4",
                f"лучший; обходит его только global LightGBM (основная модель {main_w:.3f}, +{impr:.1f}%)."]

    out += ["", "## Лог по экспериментам", "",
            "| эксперимент | результат | вердикт |", "|---|---|---|"]
    for exp, res, verdict in EXPERIMENTS:
        out.append(f"| {exp} | {res} | {verdict} |")

    main_hist = log[log["model"] == MAIN]
    if len(main_hist):
        out += ["", "## Авто-лог основной модели",
                "(дописывается при прогоне run_cv)", "",
                "| дата | commit | WRMSSE | фолдов | прогон |", "|---|---|---|---|---|"]
        for _, r in main_hist.iterrows():
            out.append(f"| {r['ts']} | {r.get('commit','?')} | {_fmt(r['WRMSSE'])} | "
                       f"{r.get('folds','')} | {r.get('note','')} |")

    MD.write_text("\n".join(out) + "\n", encoding="utf-8")


def snapshot(summary_df, folds=None, note="", ts=None, commit=None):
    """Залогировать сводку и перерисовать metriclog.md (одним вызовом из прогона)."""
    log_summary(summary_df, folds=folds, note=note, ts=ts, commit=commit)
    render()
