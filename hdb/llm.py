from __future__ import annotations

import os
from typing import Dict, List

from .config import load_config
from .utils import get_logger

try:
	from openai import OpenAI
except Exception:  # pragma: no cover
	OpenAI = None  # type: ignore


logger = get_logger("llm")


def _format_currency(x: float) -> str:
	return f"${x:,.0f}"


def _fallback_text(town: str, flat_type: str, price_bands: Dict[str, float]) -> str:
	return (
		f"In {town}, recent resale trends for {flat_type} units suggest a price range from "
		f"{_format_currency(price_bands.get('low', 0))} to {_format_currency(price_bands.get('high', 0))}, with typical transactions "
		f"around {_format_currency(price_bands.get('mid', 0))}. Prices vary by floor level, flat model, and age of the lease."
	)


def explain_prices(town: str, flat_type: str, price_bands: Dict[str, float]) -> str:
	cfg = load_config()
	api_key = os.getenv("OPENAI_API_KEY")

	prompt = (
		"You are an analyst generating brief, factual pricing insights for Singapore HDB BTO. "
		"Given a town, flat type, and 3 price bands (low/mid/high), write 2-3 sentences on market context, "
		"avoiding investment advice and speculation."
	)
	content = (
		f"Town: {town}\nFlat type: {flat_type}\n"
		f"Low: {_format_currency(price_bands.get('low', 0))}\n"
		f"Mid: {_format_currency(price_bands.get('mid', 0))}\n"
		f"High: {_format_currency(price_bands.get('high', 0))}"
	)

	if OpenAI is None or not api_key:
		return _fallback_text(town, flat_type, price_bands)

	try:
		client = OpenAI(api_key=api_key)
		resp = client.chat.completions.create(
			model=cfg.llm.model,
			messages=[
				{"role": "system", "content": prompt},
				{"role": "user", "content": content},
			],
			max_tokens=cfg.llm.max_tokens,
			temperature=cfg.llm.temperature,
		)
		return resp.choices[0].message.content or _fallback_text(town, flat_type, price_bands)
	except Exception as e:  # fallback on any API error
		logger.warning(f"LLM explain fallback due to error: {e}")
		return _fallback_text(town, flat_type, price_bands)
