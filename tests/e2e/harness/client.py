"""Thin REST client for the dendrite backend."""

from __future__ import annotations

from pathlib import Path

import httpx

from . import config


class DendriteClient:
    def __init__(self, base_url: str = config.SERVER_URL, timeout: float = 30.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DendriteClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def load_config(self, path: Path) -> dict:
        r = self._client.post("/api/config/load", params={"file_path": str(path)})
        r.raise_for_status()
        return r.json()

    def start_pipeline(self) -> dict:
        r = self._client.post("/api/pipeline/start", timeout=60.0)
        if r.status_code != 200:
            raise RuntimeError(f"Pipeline start failed [{r.status_code}]: {r.text}")
        return r.json()

    def stop_pipeline(self) -> dict:
        r = self._client.post("/api/pipeline/stop", timeout=30.0)
        r.raise_for_status()
        return r.json()

    def status(self) -> dict:
        r = self._client.get("/api/pipeline/status")
        r.raise_for_status()
        return r.json()

    def preflight(self) -> dict:
        r = self._client.get("/api/pipeline/preflight")
        r.raise_for_status()
        return r.json()
