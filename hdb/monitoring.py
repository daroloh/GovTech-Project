from __future__ import annotations

import hashlib
import os
from typing import Dict

import duckdb
import joblib
import pandas as pd

from .config import load_config


def file_hash(path: str) -> str:
	sha = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(8192), b""):
			sha.update(chunk)
	return sha.hexdigest()


def latest_data_snapshot() -> Dict:
	cfg = load_config()
	con = duckdb.connect(cfg.paths.duckdb_path, read_only=True)
	try:
		row = con.execute(
			f"SELECT COUNT(*) AS n, MIN(year) AS min_year, MAX(year) AS max_year FROM {cfg.paths.clean_table}"
		).fetchone()
		return {"rows": row[0], "min_year": row[1], "max_year": row[2]}
	finally:
		con.close()


def model_fingerprint() -> Dict:
	cfg = load_config()
	path = os.path.join(cfg.paths.model_dir, "rf_pipeline.joblib")
	if not os.path.exists(path):
		return {"exists": False}
	return {"exists": True, "sha256": file_hash(path)}
