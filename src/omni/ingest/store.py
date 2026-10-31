"""Content-addressed object store on the local filesystem (ADR 0005).

Objects are JSON documents keyed by the SHA-256 of their canonical serialisation and laid out
as ``<root>/<key[:2]>/<key>.json``. The root is ``OMNI_OBJECT_DIR`` or ``.omni/objects``.
The database never holds content, only keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omni.ingest.schema import canonical_json, sha256_hex

DEFAULT_OBJECT_DIR = Path(".omni") / "objects"


class ObjectStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("OMNI_OBJECT_DIR") or DEFAULT_OBJECT_DIR)

    @staticmethod
    def key_for(obj: dict[str, Any]) -> str:
        return sha256_hex(canonical_json(obj))

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def put(self, obj: dict[str, Any]) -> str:
        """Write ``obj`` if absent and return its key. Idempotent."""
        key = self.key_for(obj)
        path = self.path_for(key)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(canonical_json(obj), encoding="utf-8")
            tmp.replace(path)
        return key

    def get(self, key: str) -> dict[str, Any]:
        return json.loads(self.path_for(key).read_text(encoding="utf-8"))

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()
