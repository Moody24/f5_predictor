from datetime import datetime, timedelta
from pathlib import Path


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def yesterday_str() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def predictions_path(date: str = None) -> Path:
    from config.settings import PREDICTIONS_DIR
    return PREDICTIONS_DIR / f"{date or today_str()}.json"
