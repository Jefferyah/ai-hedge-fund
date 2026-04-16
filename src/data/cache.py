"""
Persistent smart cache for FinancialDatasets API responses.

Refresh strategy per data type:
  - prices:            always fresh (free tier, no cost)
  - financial_metrics:  probe with limit=1; pull full only if new report_period detected
  - search_line_items:  same probe strategy as financial_metrics
  - insider_trades:     every 7 days
  - company_news:       every 3 days
  - market_cap:         derived from free price data when possible
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Refresh intervals (seconds)
# ---------------------------------------------------------------------------
REFRESH_INTERVALS = {
    "prices": 0,                    # always fetch (free)
    "financial_metrics": 0,         # probe-based, not time-based
    "line_items": 0,                # probe-based, not time-based
    "insider_trades": 7 * 86400,    # 7 days
    "company_news": 3 * 86400,      # 3 days
}

# ---------------------------------------------------------------------------
# Disk path
# ---------------------------------------------------------------------------
_CACHE_DIR = os.environ.get(
    "SIGNALDECK_CACHE_DIR",
    os.path.join(Path.home(), ".signaldeck_cache"),
)


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _disk_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, f"{name}.json")


def _load_from_disk(name: str) -> dict | None:
    path = _disk_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load cache file %s: %s", path, e)
        return None


def _save_to_disk(name: str, data: dict):
    _ensure_cache_dir()
    path = _disk_path(name)
    try:
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    except Exception as e:
        logger.warning("Failed to save cache file %s: %s", path, e)


class Cache:
    """Persistent smart cache for API responses.

    Each cache bucket stores::

        {
            "<cache_key>": {
                "fetched_at": 1713200000.0,
                "latest_report_period": "2025-12-31",   # only for probe-based types
                "data": [ ... ]
            }
        }
    """

    DISK_NAMES = {
        "prices": "prices",
        "financial_metrics": "financial_metrics",
        "line_items": "line_items",
        "insider_trades": "insider_trades",
        "company_news": "company_news",
    }

    def __init__(self):
        # In-memory mirrors of the disk caches
        self._buckets: dict[str, dict] = {}
        self._loaded: set[str] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_bucket(self, name: str) -> dict:
        if name not in self._loaded:
            disk = _load_from_disk(name)
            self._buckets[name] = disk if disk else {}
            self._loaded.add(name)
        return self._buckets.setdefault(name, {})

    def _set_entry(self, name: str, key: str, data: list[dict], **extra):
        bucket = self._get_bucket(name)
        entry = {"fetched_at": time.time(), "data": data}
        entry.update(extra)
        bucket[key] = entry
        _save_to_disk(name, bucket)

    def _get_entry(self, name: str, key: str) -> dict | None:
        bucket = self._get_bucket(name)
        return bucket.get(key)

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """Merge existing and new data, avoiding duplicates based on a key field."""
        if not existing:
            return new_data
        existing_keys = {item.get(key_field) for item in existing if key_field in item}
        merged = existing.copy()
        merged.extend([item for item in new_data if item.get(key_field) not in existing_keys])
        return merged

    # ------------------------------------------------------------------
    # Freshness checks
    # ------------------------------------------------------------------
    def is_fresh(self, data_type: str, cache_key: str) -> bool:
        """Check whether cached data is still fresh based on refresh interval.

        For probe-based types (financial_metrics, line_items) this always
        returns False so the caller can do a probe check instead.
        """
        interval = REFRESH_INTERVALS.get(data_type, 0)
        if interval == 0:
            return False  # needs probe or always fetch
        entry = self._get_entry(data_type, cache_key)
        if not entry:
            return False
        age = time.time() - entry.get("fetched_at", 0)
        return age < interval

    def get_latest_report_period(self, data_type: str, cache_key: str) -> str | None:
        """Return the latest report_period stored for a probe-based cache entry."""
        entry = self._get_entry(data_type, cache_key)
        if not entry:
            return None
        return entry.get("latest_report_period")

    # ------------------------------------------------------------------
    # Prices (always fresh — free tier)
    # ------------------------------------------------------------------
    def get_prices(self, key: str) -> list[dict] | None:
        entry = self._get_entry("prices", key)
        return entry["data"] if entry else None

    def set_prices(self, key: str, data: list[dict]):
        self._set_entry("prices", key, data)

    # ------------------------------------------------------------------
    # Financial metrics (probe-based)
    # ------------------------------------------------------------------
    def get_financial_metrics(self, key: str) -> list[dict] | None:
        entry = self._get_entry("financial_metrics", key)
        return entry["data"] if entry else None

    def set_financial_metrics(self, key: str, data: list[dict], latest_report_period: str = None):
        extra = {}
        if latest_report_period:
            extra["latest_report_period"] = latest_report_period
        self._set_entry("financial_metrics", key, data, **extra)

    # ------------------------------------------------------------------
    # Line items (probe-based)
    # ------------------------------------------------------------------
    def get_line_items(self, key: str) -> list[dict] | None:
        entry = self._get_entry("line_items", key)
        return entry["data"] if entry else None

    def set_line_items(self, key: str, data: list[dict], latest_report_period: str = None):
        extra = {}
        if latest_report_period:
            extra["latest_report_period"] = latest_report_period
        self._set_entry("line_items", key, data, **extra)

    # ------------------------------------------------------------------
    # Insider trades (7-day refresh)
    # ------------------------------------------------------------------
    def get_insider_trades(self, key: str) -> list[dict] | None:
        entry = self._get_entry("insider_trades", key)
        return entry["data"] if entry else None

    def set_insider_trades(self, key: str, data: list[dict]):
        self._set_entry("insider_trades", key, data)

    # ------------------------------------------------------------------
    # Company news (3-day refresh)
    # ------------------------------------------------------------------
    def get_company_news(self, key: str) -> list[dict] | None:
        entry = self._get_entry("company_news", key)
        return entry["data"] if entry else None

    def set_company_news(self, key: str, data: list[dict]):
        self._set_entry("company_news", key, data)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def clear_all(self):
        """Wipe all in-memory and disk caches."""
        self._buckets.clear()
        self._loaded.clear()
        _ensure_cache_dir()
        for name in self.DISK_NAMES.values():
            path = _disk_path(name)
            if os.path.exists(path):
                os.remove(path)

    def stats(self) -> dict:
        """Return a summary of cached entries per bucket."""
        result = {}
        for name in self.DISK_NAMES.values():
            bucket = self._get_bucket(name)
            result[name] = {
                "entries": len(bucket),
                "keys": list(bucket.keys())[:10],  # show first 10
            }
        return result


# Global cache instance
_cache = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache
