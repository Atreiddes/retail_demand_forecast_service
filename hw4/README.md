# ДЗ №4. Данные для weekly demand forecast

Датасет и его описание для прототипа недельного прогноза спроса (продолжение hw2).
В качестве прокси FMCG-данных PepsiCo используется публичный M5 (Walmart, Kaggle).

## Содержимое

```
hw4/
├── data/
│   ├── raw/          сырые csv (не в git)
│   └── processed/    sales_weekly.parquet + calendar + prices (не в git)
├── notebooks/
│   └── eda.qmd       полный EDA по 13 разделам
├── scripts/
│   ├── download_m5.py   скачивание M5 (Nixtla mirror, без kaggle CLI)
│   ├── build_weekly.py  daily wide -> weekly long parquet
│   └── schemas.py       pandera-схемы для валидации
├── dataset_description.qmd   итоговый документ по 4 разделам ДЗ
├── _quarto.yml
└── README.md
```

## Источник данных

M5 Forecasting Accuracy (Walmart × University of Nicosia, Kaggle 2020).
URL соревнования: https://www.kaggle.com/competitions/m5-forecasting-accuracy
Mirror без kaggle CLI: https://github.com/Nixtla/m5-forecasts

Состав:

- 3049 уникальных артикулов × 10 магазинов × 3 штата (CA, TX, WI) = 30 490 рядов
- 1941 день истории (2011-01-29 .. 2016-06-19)
- цены по неделям (sell_prices)
- календарь с праздниками и SNAP-событиями

## Воспроизведение

```
pip install pandas pyarrow pandera matplotlib seaborn statsmodels jupytext nbformat

python scripts/download_m5.py
python scripts/build_weekly.py

quarto render notebooks/eda.qmd
quarto render dataset_description.qmd
```

Без quarto можно через jupyter:

```
jupytext --to ipynb notebooks/eda.qmd
jupyter nbconvert --to html --execute notebooks/eda.ipynb
```

## Сдаём

По постановке (слайд 20 лекции «Данные»):

1. Источник и состав данных
2. Базовый EDA с выводами, важными для моделирования
3. Оценка качества разметки и предложения по улучшению
4. Алгоритм формирования выборки и стратегия валидации

**Готовые артефакты для проверки:**

- [dataset_description.md](dataset_description.md) — итоговый документ по 4 разделам ДЗ (рендер из qmd)
- [notebooks/eda.qmd](notebooks/eda.qmd) — исходник EDA (13 разделов, рендерится в HTML через `quarto render`)
- [scripts/](scripts/) — pipeline воспроизведения: download, build_weekly, pandera-схемы

Чтобы получить HTML с графиками: `quarto render notebooks/eda.qmd` (после скачивания и сборки данных).
