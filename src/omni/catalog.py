"""Model catalog loading and database synchronization."""

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from omni.models import ModelCatalog


def load_catalog(path: str | Path = "catalog/models.yaml") -> list[dict[str, Any]]:
    """Load and validate model entries from a YAML catalog."""
    catalog_path = Path(path)

    with catalog_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("Catalog must contain a models list.")

    models = data["models"]
    seen: set[str] = set()

    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each model entry must be a mapping.")

        for field in ("model_id", "provider", "display_name", "pricing"):
            if field not in model:
                raise ValueError(f"Model is missing required field: {field}")

        model_id = model["model_id"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")

        if model_id in seen:
            raise ValueError(f"Duplicate model_id: {model_id}")

        seen.add(model_id)

        if not isinstance(model["pricing"], dict):
            raise ValueError(f"Invalid pricing for model: {model_id}")

        for key, value in model["pricing"].items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Invalid pricing.{key} for model: {model_id}")
            if value < 0:
                raise ValueError(f"Negative pricing.{key} for model: {model_id}")

    return models


def sync_catalog(session: Session, path: str | Path = "catalog/models.yaml") -> int:
    """Insert or update catalog models. Returns the number of processed models."""
    models = load_catalog(path)

    for item in models:
        model = session.get(ModelCatalog, item["model_id"])

        if model is None:
            model = ModelCatalog(model_id=item["model_id"])
            session.add(model)

        for field in (
            "provider",
            "dated_revision",
            "display_name",
            "context_window",
            "max_output",
            "capabilities",
            "pricing",
            "regions",
            "residency_class",
            "status",
            "release_date",
            "source_url",
        ):
            if field in item:
                setattr(model, field, item[field])

    session.flush()
    return len(models)
