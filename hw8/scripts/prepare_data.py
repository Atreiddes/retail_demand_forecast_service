"""Готовит данные сервиса: FOODS-срез недельной панели M5 и календарь.

Читает выход hw4 (sales_weekly.parquet, calendar.parquet) и кладёт в hw8/data только
категорию FOODS, чтобы клон был самодостаточным и лёгким. Календарь нужен на прогнозе
для snap-дней и событий будущих недель, которых нет в панели.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "hw4" / "data" / "processed"
OUT = ROOT / "data"


def main():
    OUT.mkdir(exist_ok=True)
    weekly = pd.read_parquet(SRC / "sales_weekly.parquet")
    foods = weekly[weekly["cat_id"] == "FOODS"].copy()
    foods.to_parquet(OUT / "foods_weekly.parquet")
    pd.read_parquet(SRC / "calendar.parquet").to_parquet(OUT / "calendar.parquet")
    print(f"FOODS: {foods['id'].nunique():,} рядов, {len(foods):,} строк -> data/foods_weekly.parquet")
    print("календарь -> data/calendar.parquet")


if __name__ == "__main__":
    main()
