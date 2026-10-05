"""Download the Kaggle 'Customer Support on Twitter' dataset.

Usage: python -m src.download
Needs Kaggle creds: either ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY
in .env (we export them to the env vars the kaggle client reads).
"""
import os
import sys
from pathlib import Path
from src.common import config, ROOT

DATASET = "thoughtvector/customer-support-on-twitter"


def main():
    cfg = config()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    target = raw_dir / "twcs.csv"
    if target.exists():
        print(f"Already downloaded: {target}")
        return

    # kaggle.json takes priority; otherwise fall back to env vars from .env
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            sys.exit(
                "No Kaggle credentials. Put kaggle.json at ~/.kaggle/kaggle.json "
                "or set KAGGLE_USERNAME and KAGGLE_KEY in .env"
            )

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} to {raw_dir} ...")
    api.dataset_download_files(DATASET, path=str(raw_dir), unzip=True)
    print(f"Done. Files: {[p.name for p in raw_dir.iterdir()]}")


if __name__ == "__main__":
    main()
