from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol, Optional
import requests

from typing import Optional, Dict, Any


@dataclass
class OllamaProvider:
    # model: str = "llama3.1:8b"
    # base_url: str = "http://localhost:11434"
    model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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
    
    
@dataclass
class OpenAIProvider:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
        from openai import OpenAI

        client = OpenAI()

        kwargs = dict(
            model=self.model,
            input=prompt,
            temperature=0,
            store=False,  # good for benchmarking privacy stance
        )

        # Structured Outputs via Responses API: text.format = {type: "json_schema", ...}
        if json_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "schema": json_schema["schema"],
                    "strict": True,
                }
            }

        resp = client.responses.create(**kwargs)
        return (resp.output_text or "").strip()


# @dataclass
# class OpenAIProvider:
#     model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # choose your default
#     # You can also set OPENAI_BASE_URL for proxies, but usually not needed.

#     def generate(self, prompt: str) -> str:
#         from openai import OpenAI

#         client = OpenAI()  # reads OPENAI_API_KEY from env

#         resp = client.responses.create(
#             model=self.model,
#             input=prompt,
#             # Keep behavior comparable across models
#             temperature=0,
#         )

#         # Responses API returns content items; easiest is "output_text"
#         return (resp.output_text or "").strip()
