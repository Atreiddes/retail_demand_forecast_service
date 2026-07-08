# MFDP: weekly demand forecast (M5)

Учебный проект MFDP 2026. Прогноз недельного спроса на открытых данных M5 Walmart как прокси
FMCG (PepsiCo). Из четырёх задач первоначального анализа в прототип и baseline вынесена одна,
прогноз спроса; остальные три в roadmap.

## Карта работ

| Этап | Что | Где |
|---|---|---|
| hw1 | Бизнес-анализ: 4 задачи на M5 (прогноз, эластичность, promo uplift, оптимизация цен) | [hw1/business_analysis.md](hw1/business_analysis.md) |
| hw2 | Прототип weekly demand forecast для demand planner-а, сужение scope, SaaS-дизайн | [hw2/prototype.md](hw2/prototype.md) |
| hw3 | Презентация и расчёты (Google Slides / Sheets) | [hw3/](hw3/) |
| hw4 | Датасет M5 + EDA для weekly demand forecast | [hw4/](hw4/) |
| hw5-6 | Baseline-решение: модели, метрики, сервис | [hw5/](hw5/) |

## Главный результат (hw5-6)

Недельный прогноз спроса item × store на 4 недели, 30490 рядов. Лучшая модель LightGBM ансамбль,
WRMSSE 0.748 (8 фолдов walk-forward), обходит простой MA-4 (0.802) на 6.8%. Лестница моделей от
наива до бустинга в [hw5/metriclog.md](hw5/metriclog.md), разбор метрик в
[hw5/evaluation.md](hw5/evaluation.md). Точка входа в решение: [hw5/README.md](hw5/README.md).

## Сужение scope

hw1 ставил четыре задачи на дневном горизонте. В hw2 для пилота с demand planner-ом оставлена
одна, прогноз спроса (недельный горизонт, понятный планировщику KPI). Эластичность, promo uplift
и оптимизация цен надстраиваются над прогнозом и вынесены в roadmap, для baseline они не нужны.

Dmitrii Gertsovskii.
