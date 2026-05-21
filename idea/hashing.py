"""Stable hashing helpers for IDEA artifact caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any, length: int = 16) -> str:
    digest = hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()
    return digest[:length]
