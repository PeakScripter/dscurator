from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from litellm import completion


class LLMProvider:
	"""Unified LLM caller using LiteLLM so we work with many services.

	Supported through LiteLLM: OpenAI, Anthropic, Google, Cohere, Groq, Ollama, etc.
	Users pass any `model` string compatible with LiteLLM.
	"""

	def __init__(self, model: str, api_base: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None) -> None:
		load_dotenv()
		self.model = model
		self.api_base = api_base
		self.extra_headers = extra_headers or {}

	def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
		messages: List[Dict[str, str]] = []
		if system:
			messages.append({"role": "system", "content": system})
		messages.append({"role": "user", "content": prompt})

		kwargs: Dict[str, object] = {
			"model": self.model,
			"messages": messages,
			"temperature": temperature,
		}
		if max_tokens is not None:
			kwargs["max_tokens"] = max_tokens
		if self.api_base:
			kwargs["api_base"] = self.api_base
		if self.extra_headers:
			kwargs["extra_headers"] = self.extra_headers

		resp = completion(**kwargs)  # type: ignore[arg-type]
		return resp["choices"][0]["message"]["content"].strip()


