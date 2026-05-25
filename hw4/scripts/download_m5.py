"""Скачивает m5.zip из Nixtla mirror в hw4/data/raw/ и распаковывает."""
from pathlib import Path
import sys
import urllib.request
import zipfile

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED = [
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "calendar.csv",
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    if all((RAW / f).exists() for f in EXPECTED):
        print(f"all M5 files in {RAW}, skip")
        return

    zip_path = RAW / "m5.zip"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    print(f"  {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("extracting")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    zip_path.unlink()

    missing = [f for f in EXPECTED if not (RAW / f).exists()]
    if missing:
        print(f"WARN: missing after unzip: {missing}", file=sys.stderr)
        print("contents of raw/:")
        for p in sorted(RAW.iterdir()):
            print(f"  {p.name}")
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
