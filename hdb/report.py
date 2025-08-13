from __future__ import annotations

import os
from typing import List, Dict, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .llm import explain_prices
from .utils import get_logger, utc_now_str


logger = get_logger("report")


def _fmt_currency(x: float) -> str:
	return f"${x:,.0f}"


def _income_needed(price: float, ratio: float = 0.3) -> float:
	years = 5
	return price / (years * 12 * ratio)


def _default_area(flat_type: str) -> float:
	ft = (flat_type or "").upper().strip()
	if ft == "3 ROOM":
		return 65.0
	if ft == "4 ROOM":
		return 95.0
	return 80.0


def _load_pipeline(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError("Model not trained yet. Run training first.")
	return joblib.load(path)


def _recommend_towns(limit: int, flat_types: List[str]) -> List[str]:
	cfg = load_config()
	con = duckdb.connect(cfg.paths.duckdb_path, read_only=True)
	try:
		rows = con.execute(
			f"""
			WITH recent AS (
				SELECT town, flat_type, COUNT(*) AS n
				FROM {cfg.paths.clean_table}
				WHERE year >= 2017
				GROUP BY town, flat_type
			), ranked AS (
				SELECT town, SUM(n) AS total_recent
				FROM recent
				WHERE flat_type IN ({', '.join(['?' for _ in flat_types])})
				GROUP BY town
				ORDER BY total_recent ASC NULLS FIRST
			)
			SELECT town FROM ranked LIMIT ?
			""",
			flat_types + [limit],
		).fetchall()
		return [r[0] for r in rows]
	finally:
		con.close()


def generate_bto_report(
	towns: List[str] | None = None,
	flat_types: List[str] | None = None,
	low_floor: float = 5,
	mid_floor: float = 12,
	high_floor: float = 25,
	limit_if_recommend: int = 5,
	output_path: str = "artifacts/bto_report.md",
) -> str:
	cfg = load_config()
	flat_types = flat_types or ["3 ROOM", "4 ROOM"]
	if towns is None or len(towns) == 0:
		towns = _recommend_towns(limit_if_recommend, flat_types)
		logger.info(f"Auto-selected towns: {towns}")

	pipe = _load_pipeline(os.path.join(cfg.paths.model_dir, "rf_pipeline.joblib"))

	# Load median areas
	con = duckdb.connect(cfg.paths.duckdb_path, read_only=True)
	try:
		areas = con.execute(
			f"""
			SELECT town, flat_type, median(floor_area_sqm) AS med_area
			FROM {cfg.paths.features_table}
			GROUP BY town, flat_type
			"""
		).df()
	finally:
		con.close()

	rows: List[Dict] = []
	for town in towns:
		for ft in flat_types:
			# choose area
			match = areas[(areas["town"] == town) & (areas["flat_type"] == ft)]
			if match.empty:
				area = _default_area(ft)
			else:
				val = match["med_area"].iloc[0]
				med = float(val) if pd.notna(val) else float("nan")
				area = med if not np.isnan(med) else _default_area(ft)

			for label, floor in [("low", low_floor), ("mid", mid_floor), ("high", high_floor)]:
				rows.append({
					"town": town,
					"flat_type": ft,
					"flat_model": "Improved",
					"floor_area_sqm": float(area),
					"lease_commence_date": 1990,
					"storey_mid": float(floor),
					"year": 2023,
					"month_num": 6,
					"_band": label,
				})

	df = pd.DataFrame(rows)
	preds = pipe.predict(df.drop(columns=["_band"]))
	df["predicted_resale_price"] = preds
	disc = cfg.training.discount_rate
	df["bto_price"] = df["predicted_resale_price"] * (1 - disc)
	df["income"] = df["bto_price"].map(lambda x: _income_needed(float(x)))

	# Build markdown
	lines: List[str] = []
	lines.append("# BTO Recommendations and Price Analysis")
	lines.append("")
	lines.append(f"Generated: {utc_now_str()}")
	lines.append("")
	lines.append(f"Discount rate applied to resale predictions: {int(disc*100)}%")
	lines.append("")

	for town in towns:
		lines.append(f"## {town}")
		for ft in flat_types:
			g = df[(df["town"] == town) & (df["flat_type"] == ft)]
			if g.empty:
				continue
			bands = {r["_band"]: float(r["bto_price"]) for _, r in g.iterrows()}
			inc = {r["_band"]: float(r["income"]) for _, r in g.iterrows()}
			expl = explain_prices(town, ft, {
				"low": bands.get("low", 0.0),
				"mid": bands.get("mid", 0.0),
				"high": bands.get("high", 0.0),
			})
			lines.append(f"- {ft}")
			lines.append(
				f"  - Prices: low {_fmt_currency(bands.get('low', 0))}, mid {_fmt_currency(bands.get('mid', 0))}, high {_fmt_currency(bands.get('high', 0))}"
			)
			lines.append(
				f"  - Incomes: low {_fmt_currency(inc.get('low', 0))}/mo, mid {_fmt_currency(inc.get('mid', 0))}/mo, high {_fmt_currency(inc.get('high', 0))}/mo"
			)
			lines.append(f"  - Note: {expl}")
		lines.append("")

	md = "\n".join(lines)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(md)
	logger.info(f"Report written to {output_path}")
	return md
