from __future__ import annotations

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_config


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
	categorical = ["town", "flat_type", "flat_model"]
	numeric = ["floor_area_sqm", "lease_commence_date", "storey_mid", "year", "month_num"]
	preprocessor = ColumnTransformer(
		transformers=[
			("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
			("num", StandardScaler(), numeric),
		]
	)
	return preprocessor


def load_training_dataframe() -> pd.DataFrame:
	cfg = load_config()
	con = duckdb.connect(cfg.paths.duckdb_path, read_only=True)
	try:
		df = con.execute(f"SELECT * FROM {cfg.paths.features_table}").df()
		return df
	finally:
		con.close()
