from __future__ import annotations

import glob
import os
from typing import List

import duckdb
import pandas as pd

from .config import load_config
from .utils import get_logger


logger = get_logger("etl")


def detect_schema_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
	cols = {c.lower().strip().replace(" ", "_") for c in df.columns}
	# unify common columns across the provided HDB datasets
	column_mapping = {}
	if "month" in cols:
		column_mapping[[c for c in df.columns if c.lower() == "month"][0]] = "month"
	if "town" in cols:
		column_mapping[[c for c in df.columns if c.lower() == "town"][0]] = "town"
	if "flat_type" in cols:
		column_mapping[[c for c in df.columns if c.lower() == "flat_type"][0]] = "flat_type"
	if "flat_model" in cols:
		column_mapping[[c for c in df.columns if c.lower() == "flat_model"][0]] = "flat_model"
	# floor related
	floor_map_keys = [c for c in df.columns if c.lower().replace(" ", "_") in ("storey_range", "floor_range")]
	if floor_map_keys:
		column_mapping[floor_map_keys[0]] = "storey_range"
	# block/street
	for src, tgt in [("block", "block"), ("street_name", "street_name"), ("street", "street_name")]:
		match = [c for c in df.columns if c.lower().replace(" ", "_") == src]
		if match:
			column_mapping[match[0]] = tgt
	# area
	for src in ("floor_area_sqm", "area_sqm"):
		match = [c for c in df.columns if c.lower().replace(" ", "_") == src]
		if match:
			column_mapping[match[0]] = "floor_area_sqm"
			break
	# lease/remaining lease
	lease_match = [c for c in df.columns if c.lower().replace(" ", "_") in ("lease_commence_date", "lease_commence")]
	if lease_match:
		column_mapping[lease_match[0]] = "lease_commence_date"
	# transaction date and price
	for src in ("resale_price", "price"):
		match = [c for c in df.columns if c.lower().replace(" ", "_") == src]
		if match:
			column_mapping[match[0]] = "resale_price"
			break

	for src in ("month", "transaction_month", "date"):
		match = [c for c in df.columns if c.lower().replace(" ", "_") == src]
		if match:
			column_mapping[match[0]] = "month"
			break

	df = df.rename(columns=column_mapping)
	# keep only known columns
	keep = [
		"month",
		"town",
		"flat_type",
		"flat_model",
		"storey_range",
		"block",
		"street_name",
		"floor_area_sqm",
		"lease_commence_date",
		"resale_price",
	]
	present = [c for c in keep if c in df.columns]
	out = df[present].copy()
	for c in keep:
		if c not in out.columns:
			out[c] = pd.NA
	# reorder columns
	out = out[keep]
	return out


def parse_storey_midpoint(storey_range: str) -> float:
	if not isinstance(storey_range, str):
		return float("nan")
	try:
		s = storey_range.strip().upper().replace(" TO ", "-")
		s = s.replace(" TO", "-").replace("TO ", "-")
		parts = s.split("-")
		low = int(parts[0])
		high = int(parts[1]) if len(parts) > 1 else low
		return (low + high) / 2.0
	except Exception:
		return float("nan")


def load_csvs_to_duckdb(csv_paths: List[str] | None = None) -> None:
	cfg = load_config()
	conn = duckdb.connect(cfg.paths.duckdb_path)
	try:
		logger.info("Creating raw table if not exists")
		conn.execute(
			f"""
			CREATE TABLE IF NOT EXISTS {cfg.paths.raw_table} (
				source_file TEXT,
				month TEXT,
				town TEXT,
				flat_type TEXT,
				flat_model TEXT,
				storey_range TEXT,
				block TEXT,
				street_name TEXT,
				floor_area_sqm DOUBLE,
				lease_commence_date INTEGER,
				resale_price DOUBLE
			);
			"""
		)

		if csv_paths is None:
			csv_paths = [p for p in glob.glob("*.csv")]
		# clear existing to avoid duplicates on reruns
		conn.execute(f"DELETE FROM {cfg.paths.raw_table}")
		all_rows = 0
		for path in csv_paths:
			logger.info(f"Reading {path}")
			df = pd.read_csv(path)
			df = detect_schema_and_standardize(df)
			df["source_file"] = os.path.basename(path)
			# clean types
			if "resale_price" in df.columns:
				df["resale_price"] = (
					df["resale_price"].astype(str).str.replace(",", "", regex=False).astype(float)
				)
			if "floor_area_sqm" in df.columns:
				df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
			if "lease_commence_date" in df.columns:
				df["lease_commence_date"] = pd.to_numeric(
					df["lease_commence_date"], errors="coerce"
				).astype("Int64")

			# register df and insert with explicit column ordering to avoid misalignment
			conn.register("df", df)
			conn.execute(
				f"""
				INSERT INTO {cfg.paths.raw_table}
				(source_file, month, town, flat_type, flat_model, storey_range, block, street_name, floor_area_sqm, lease_commence_date, resale_price)
				SELECT source_file, month, town, flat_type, flat_model, storey_range, block, street_name, floor_area_sqm, lease_commence_date, resale_price
				FROM df
				"""
			)
			all_rows += len(df)
		logger.info(f"Inserted {all_rows} raw rows")

		# create clean table
		logger.info("Creating clean table")
		conn.execute(
			f"""
			CREATE OR REPLACE TABLE {cfg.paths.clean_table} AS
			SELECT
				*,
				CAST(strptime(month || '-01', '%Y-%m-%d') AS DATE) AS txn_date,
				EXTRACT(year FROM CAST(strptime(month || '-01', '%Y-%m-%d') AS DATE)) AS year,
				EXTRACT(month FROM CAST(strptime(month || '-01', '%Y-%m-%d') AS DATE)) AS month_num
			FROM {cfg.paths.raw_table}
			WHERE resale_price IS NOT NULL AND town IS NOT NULL AND flat_type IS NOT NULL;
			"""
		)

		# add features table by computing storey_mid in pandas and writing back
		logger.info("Creating features table")
		feat_df = conn.execute(
			f"""
			SELECT resale_price, town, flat_type, flat_model, floor_area_sqm,
			       lease_commence_date, storey_range, year, month_num
			FROM {cfg.paths.clean_table}
			WHERE resale_price > 10000 AND floor_area_sqm IS NOT NULL
			"""
		).df()
		feat_df["storey_mid"] = feat_df["storey_range"].map(parse_storey_midpoint)
		feat_df = feat_df.drop(columns=["storey_range"])
		conn.register("feat_df", feat_df)
		conn.execute(f"CREATE OR REPLACE TABLE {cfg.paths.features_table} AS SELECT * FROM feat_df")
	finally:
		conn.close()


if __name__ == "__main__":
	load_csvs_to_duckdb()
