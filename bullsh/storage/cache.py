"""User-controlled caching for API responses and scraped data."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bullsh.config import get_config


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    key: str
    data: dict[str, Any]
    source: str  # Tool that produced this data
    ticker: str | None
    created_at: str  # ISO format
    expires_at: str | None  # ISO format, None = never expires
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > datetime.fromisoformat(self.expires_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(**data)


# Default TTLs by data source (in hours)
DEFAULT_TTL: dict[str, int | None] = {
    "sec": 24 * 7,      # SEC filings - 7 days (rarely change)
    "yahoo": 1,         # Yahoo Finance - 1 hour (price/ratings change)
    "stocktwits": 1,    # StockTwits - 1 hour (sentiment is time-sensitive)
    "reddit": 2,        # Reddit - 2 hours
    "news": 4,          # News - 4 hours
    "price_history": 24,    # Price history - 1 day (EOD refresh)
    "fama_french": 24 * 7,  # Fama-French factors - 7 days (monthly updates)
    "default": 12,      # Default - 12 hours
}


class Cache:
    """File-based cache with user-controlled refresh."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, CacheEntry] = {}
        self._load_index()

    def _index_path(self) -> Path:
        return self.cache_dir / "index.json"

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                    self._index = {
                        k: CacheEntry.from_dict(v)
                        for k, v in data.items()
                    }
            except (json.JSONDecodeError, KeyError):
                self._index = {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        with open(self._index_path(), "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._index.items()},
                f,
                indent=2,
            )

    def _make_key(self, source: str, identifier: str, **params: Any) -> str:
        """Create a cache key from source and parameters."""
        key_parts = [source, identifier]
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")
        key_string = ":".join(str(p) for p in key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _data_path(self, key: str) -> Path:
        """Get the file path for cached data."""
        return self.cache_dir / f"{key}.json"

    def get(
        self,
        source: str,
        identifier: str,
        **params: Any,
    ) -> dict[str, Any] | None:
        """
        Retrieve cached data if it exists and hasn't expired.

        Args:
            source: Tool name (sec, yahoo, etc.)
            identifier: Primary identifier (usually ticker)
            **params: Additional parameters for cache key

        Returns:
            Cached data or None if not found/expired
        """
        key = self._make_key(source, identifier, **params)

        if key not in self._index:
            return None

        entry = self._index[key]
        if entry.is_expired():
            self.invalidate(source, identifier, **params)
            return None

        # Load data from file
        data_path = self._data_path(key)
        if not data_path.exists():
            del self._index[key]
            self._save_index()
            return None

        try:
            with open(data_path) as f:
                data = json.load(f)

            # Update hit count
            entry.hit_count += 1
            self._save_index()

            return data
        except json.JSONDecodeError:
            self.invalidate(source, identifier, **params)
            return None

    def set(
        self,
        source: str,
        identifier: str,
        data: dict[str, Any],
        ttl_hours: int | None = None,
        **params: Any,
    ) -> None:
        """
        Cache data with optional TTL.

        Args:
            source: Tool name
            identifier: Primary identifier (usually ticker)
            data: Data to cache
            ttl_hours: Time to live in hours (None = use default for source)
            **params: Additional parameters for cache key
        """
        key = self._make_key(source, identifier, **params)

        # Determine TTL
        if ttl_hours is None:
            ttl_hours = DEFAULT_TTL.get(source, DEFAULT_TTL["default"])

        now = datetime.now()
        expires_at = None
        if ttl_hours is not None:
            expires_at = (now + timedelta(hours=ttl_hours)).isoformat()

        # Create entry
        entry = CacheEntry(
            key=key,
            data={},  # Stored in separate file
            source=source,
            ticker=identifier if source != "news" else None,
            created_at=now.isoformat(),
            expires_at=expires_at,
        )

        # Save data to file
        data_path = self._data_path(key)
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Update index
        self._index[key] = entry
        self._save_index()

    def invalidate(
        self,
        source: str,
        identifier: str,
        **params: Any,
    ) -> bool:
        """
        Invalidate a specific cache entry.

        Returns:
            True if entry was found and removed
        """
        key = self._make_key(source, identifier, **params)

        if key not in self._index:
            return False

        # Remove data file
        data_path = self._data_path(key)
        if data_path.exists():
            data_path.unlink()

        # Remove from index
        del self._index[key]
        self._save_index()
        return True

    def invalidate_ticker(self, ticker: str) -> int:
        """
        Invalidate all cache entries for a ticker.

        Args:
            ticker: Stock ticker to invalidate

        Returns:
            Number of entries invalidated
        """
        to_remove = [
            key for key, entry in self._index.items()
            if entry.ticker and entry.ticker.upper() == ticker.upper()
        ]

        for key in to_remove:
            data_path = self._data_path(key)
            if data_path.exists():
                data_path.unlink()
            del self._index[key]

        if to_remove:
            self._save_index()

        return len(to_remove)

    def invalidate_source(self, source: str) -> int:
        """
        Invalidate all cache entries from a specific source.

        Args:
            source: Tool name to invalidate

        Returns:
            Number of entries invalidated
        """
        to_remove = [
            key for key, entry in self._index.items()
            if entry.source == source
        ]

        for key in to_remove:
            data_path = self._data_path(key)
            if data_path.exists():
                data_path.unlink()
            del self._index[key]

        if to_remove:
            self._save_index()

        return len(to_remove)

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of entries cleared
        """
        count = len(self._index)

        # Remove all data files
        for key in self._index:
            data_path = self._data_path(key)
            if data_path.exists():
                data_path.unlink()

        # Clear index
        self._index = {}
        self._save_index()

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        by_source: dict[str, int] = {}
        expired_count = 0

        for entry in self._index.values():
            data_path = self._data_path(entry.key)
            if data_path.exists():
                size = data_path.stat().st_size
                total_size += size
                by_source[entry.source] = by_source.get(entry.source, 0) + 1

            if entry.is_expired():
                expired_count += 1

        return {
            "total_entries": len(self._index),
            "expired_entries": expired_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_source": by_source,
        }

    def list_entries(
        self,
        source: str | None = None,
        ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """List cache entries with optional filtering."""
        entries = []
        for entry in self._index.values():
            if source and entry.source != source:
                continue
            if ticker and entry.ticker != ticker.upper():
                continue

            entries.append({
                "key": entry.key,
                "source": entry.source,
                "ticker": entry.ticker,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "expired": entry.is_expired(),
                "hit_count": entry.hit_count,
            })

        return sorted(entries, key=lambda x: x["created_at"], reverse=True)


# Global cache instance (lazy loaded)
_cache: Cache | None = None


def get_cache() -> Cache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def reset_cache() -> None:
    """Reset the global cache instance (useful for testing)."""
    global _cache
    _cache = None
