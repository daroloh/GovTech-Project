from __future__ import annotations

import os
from typing import Dict, List, Optional

import duckdb
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .config import load_config
from .llm import explain_prices
from .utils import get_logger
from .report import generate_bto_report


logger = get_logger("api")
app = FastAPI(title="HDB BTO Pricing API", version="0.1.0")
_cfg = load_config()


def _default_area(flat_type: str) -> float:
	ft = (flat_type or "").upper().strip()
	if ft == "3 ROOM":
		return 65.0
	if ft == "4 ROOM":
		return 95.0
	return 80.0


class PredictRequest(BaseModel):
	town: str
	flat_type: str
	floor_area_sqm: float
	storey_mid: float
	flat_model: Optional[str] = None
	lease_commence_date: Optional[int] = None
	year: Optional[int] = 2023
	month_num: Optional[int] = 6


class PredictResponse(BaseModel):
	predicted_resale_price: float
	bto_price_low: float
	bto_price_mid: float
	bto_price_high: float
	income_low: float
	income_mid: float
	income_high: float
	explanation: str


class BTOAnalysisRequest(BaseModel):
	towns: List[str]
	flat_types: List[str] = ["3 ROOM", "4 ROOM"]
	low_floor: float = 5
	mid_floor: float = 12
	high_floor: float = 25
	floor_area_sqm: Optional[float] = None  # if None, use town+type median area


MODEL_PATH = os.path.join(_cfg.paths.model_dir, "rf_pipeline.joblib")


def _load_pipeline():
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError("Model not trained yet. Run training first.")
	return joblib.load(MODEL_PATH)


def _income_needed(price: float, ratio: float = 0.3) -> float:
	years = 5
	return price / (years * 12 * ratio)


@app.get("/health")
def health():
	return {"status": "ok"}


@app.get("/metrics")
def metrics():
	try:
		with open(_cfg.paths.metrics_path, "r", encoding="utf-8") as f:
			import json
			return json.load(f)
	except FileNotFoundError:
		raise HTTPException(404, detail="Metrics not found. Train the model first.")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
	pipe = _load_pipeline()
	row = {
		"town": req.town,
		"flat_type": req.flat_type,
		"flat_model": req.flat_model or "Improved",
		"floor_area_sqm": req.floor_area_sqm,
		"lease_commence_date": req.lease_commence_date or 1990,
		"storey_mid": req.storey_mid,
		"year": req.year or 2023,
		"month_num": req.month_num or 6,
	}
	import pandas as pd
	df = pd.DataFrame([row])
	pred = float(pipe.predict(df)[0])

	disc = _cfg.training.discount_rate
	low = pred * (1 - disc * 1.1)
	mid = pred * (1 - disc)
	high = pred * (1 - disc * 0.9)

	income_low = _income_needed(low)
	income_mid = _income_needed(mid)
	income_high = _income_needed(high)

	expl = explain_prices(req.town, req.flat_type, {"low": low, "mid": mid, "high": high})

	return PredictResponse(
		predicted_resale_price=pred,
		bto_price_low=low,
		bto_price_mid=mid,
		bto_price_high=high,
		income_low=income_low,
		income_mid=income_mid,
		income_high=income_high,
		explanation=expl,
	)


@app.get("/recommend")
def recommend(
	limit: int = Query(5, ge=1, le=20),
	flat_types: List[str] = Query(["3 ROOM", "4 ROOM"]),
):
	con = duckdb.connect(_cfg.paths.duckdb_path, read_only=True)
	try:
		result = con.execute(
			f"""
			WITH recent AS (
				SELECT town, flat_type, COUNT(*) AS n
				FROM {_cfg.paths.clean_table}
				WHERE year >= 2017
				GROUP BY town, flat_type
			), ranked AS (
				SELECT town, SUM(n) AS total_recent
				FROM recent
				WHERE flat_type IN ({', '.join(['?' for _ in flat_types])})
				GROUP BY town
				ORDER BY total_recent ASC NULLS FIRST
			)
			SELECT town, total_recent FROM ranked LIMIT ?
			""",
			flat_types + [limit],
		).fetchall()
	finally:
		con.close()

	return {"limit": limit, "flat_types": flat_types, "towns": result}


@app.post("/bto_analysis")
def bto_analysis(req: BTOAnalysisRequest):
	try:
		pipe = _load_pipeline()
		import pandas as pd
		con = duckdb.connect(_cfg.paths.duckdb_path, read_only=True)
		try:
			areas = con.execute(
				f"""
				SELECT town, flat_type, median(floor_area_sqm) AS med_area
				FROM {_cfg.paths.features_table}
				GROUP BY town, flat_type
				"""
			).df()
		finally:
			con.close()

		rows = []
		for town in req.towns:
			for ft in req.flat_types:
				area = req.floor_area_sqm
				if area is None:
					match = areas[(areas["town"] == town) & (areas["flat_type"] == ft)]
					if match.empty:
						area = _default_area(ft)
					else:
						med = float(match["med_area"].iloc[0]) if match["med_area"].notna().iloc[0] else float("nan")
						area = med if not np.isnan(med) else _default_area(ft)

				for label, floor in [("low", req.low_floor), ("mid", req.mid_floor), ("high", req.high_floor)]:
					row = {
						"town": town,
						"flat_type": ft,
						"flat_model": "Improved",
						"floor_area_sqm": float(area),
						"lease_commence_date": 1990,
						"storey_mid": float(floor),
						"year": 2023,
						"month_num": 6,
						"_band": label,
					}
					rows.append(row)

		df = pd.DataFrame(rows)
		preds = pipe.predict(df.drop(columns=["_band"]))
		df["predicted_resale_price"] = preds
		disc = _cfg.training.discount_rate
		df["bto_price"] = df["predicted_resale_price"] * (1 - disc)
		df["income"] = df["bto_price"].map(lambda x: _income_needed(float(x)))

		# aggregate by town+flat_type
		out = []
		for (town, ft), g in df.groupby(["town", "flat_type"]):
			bands = {r["_band"]: float(r["bto_price"]) for _, r in g.iterrows()}
			expl = explain_prices(town, ft, {
				"low": bands.get("low", 0.0),
				"mid": bands.get("mid", 0.0),
				"high": bands.get("high", 0.0),
			})
			out.append({
				"town": town,
				"flat_type": ft,
				"bto_prices": bands,
				"income": {
					"low": float(g[g["_band"] == "low"]["income"].iloc[0]),
					"mid": float(g[g["_band"] == "mid"]["income"].iloc[0]),
					"high": float(g[g["_band"] == "high"]["income"].iloc[0]),
				},
				"explanation": expl,
			})

		return {"results": out}
	except Exception as e:
		logger.exception("bto_analysis failed")
		raise HTTPException(status_code=500, detail=f"bto_analysis error: {e}")


@app.get("/report_md")

def report_md(
	towns: Optional[str] = Query(None, description="Comma-separated towns; if omitted, auto-recommend"),
	flat_types: str = Query("3 ROOM,4 ROOM"),
	low_floor: float = 5,
	mid_floor: float = 12,
	high_floor: float = 25,
	limit: int = 5,
):
	try:
		town_list = [t.strip() for t in towns.split(",")] if towns else None
		ft_list = [t.strip() for t in flat_types.split(",")]
		md = generate_bto_report(
			towns=town_list,
			flat_types=ft_list,
			low_floor=low_floor,
			mid_floor=mid_floor,
			high_floor=high_floor,
			limit_if_recommend=limit,
			output_path="artifacts/bto_report.md",
		)
		return {"markdown": md}
	except Exception as e:
		logger.exception("report_md failed")
		raise HTTPException(status_code=500, detail=f"report_md error: {e}")
