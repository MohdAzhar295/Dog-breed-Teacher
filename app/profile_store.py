from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import BREED_PROFILES_PATH, load_selected_breeds


def _slugify(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


class ProfileStore:
    def __init__(self) -> None:
        self._profiles_by_id: dict[str, dict[str, Any]] = {}
        self._alias_to_id: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        with BREED_PROFILES_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        for profile in payload["breeds"]:
            breed_id = profile["breed_id"]
            self._profiles_by_id[breed_id] = profile
            aliases = {breed_id, profile["display_name"], *profile.get("aliases", [])}
            for alias in aliases:
                self._alias_to_id[_slugify(alias)] = breed_id

        for breed in load_selected_breeds():
            breed_id = breed["id"]
            aliases = {breed_id, breed["display_name"], *breed.get("aliases", [])}
            for alias in aliases:
                self._alias_to_id[_slugify(alias)] = breed_id

    def list_breeds(self) -> list[dict[str, Any]]:
        return [
            {
                "breed_id": profile["breed_id"],
                "display_name": profile["display_name"],
                "temperament_tags": profile.get("temperament_tags", []),
                "origin_country": (profile.get("origin") or {}).get("origin_country"),
            }
            for profile in self._profiles_by_id.values()
        ]

    def get_profile(self, breed_key: str) -> dict[str, Any] | None:
        breed_id = self._alias_to_id.get(_slugify(breed_key), breed_key)
        return self._profiles_by_id.get(breed_id)

    def merge_with_api_data(self, profile: dict[str, Any], api_data: dict[str, Any] | None) -> dict[str, Any]:
        if not api_data:
            return profile

        merged = dict(profile)
        merged["api_enriched"] = True

        if api_data.get("temperament"):
            merged["temperament_summary"] = api_data["temperament"]
        if api_data.get("bred_for"):
            merged["bred_for"] = api_data["bred_for"]
        if api_data.get("breed_group"):
            merged["breed_group"] = api_data["breed_group"]
        if api_data.get("origin"):
            origin = merged.get("origin", {})
            if isinstance(origin, dict):
                origin = dict(origin)
                origin["api_origin"] = api_data["origin"]
                merged["origin"] = origin
            else:
                merged["origin"] = {"api_origin": api_data["origin"]}
        if api_data.get("life_span"):
            merged["life_span"] = api_data["life_span"]
        if api_data.get("weight"):
            merged["weight"] = api_data["weight"]
        if api_data.get("height"):
            merged["height"] = api_data["height"]

        return merged
