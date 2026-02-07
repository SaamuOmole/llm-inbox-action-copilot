from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Optional
import requests


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class OllamaProvider:
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
