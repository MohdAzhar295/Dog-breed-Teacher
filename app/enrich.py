from __future__ import annotations

import os
from typing import Any

import httpx

from app.profile_store import ProfileStore

THEDOGAPI_URL = "https://api.thedogapi.com/v1/breeds"


class BreedEnricher:
    def __init__(self, profile_store: ProfileStore) -> None:
        self.profile_store = profile_store
        self.api_key = os.getenv("THEDOGAPI_KEY", "").strip()
        self._cache: dict[str, dict[str, Any] | None] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _match_breed(self, breeds: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any] | None:
        candidates = {
            profile["display_name"].lower(),
            profile["breed_id"].replace("_", " ").lower(),
            *[alias.lower() for alias in profile.get("aliases", [])],
        }

        for breed in breeds:
            names = {breed.get("name", "").lower()}
            names.update(alias.lower() for alias in breed.get("alt_names", "").split(",") if alias.strip())
            if candidates & names:
                return breed
        return None

    async def enrich(self, breed_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        if breed_id in self._cache:
            return self._cache[breed_id]

        profile = self.profile_store.get_profile(breed_id)
        if not profile:
            self._cache[breed_id] = None
            return None

        headers = {"x-api-key": self.api_key}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(THEDOGAPI_URL, headers=headers)
            response.raise_for_status()
            breeds = response.json()

        matched = self._match_breed(breeds, profile)
        self._cache[breed_id] = matched
        return matched
