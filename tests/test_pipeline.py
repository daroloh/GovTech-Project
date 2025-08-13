import os
import json

from hdb.config import load_config


def test_artifacts_exist():
	cfg = load_config()
	assert os.path.exists(cfg.paths.duckdb_path)
	assert os.path.exists(cfg.paths.model_dir)
	metrics_path = cfg.paths.metrics_path
	assert os.path.exists(metrics_path)
	with open(metrics_path, "r", encoding="utf-8") as f:
		m = json.load(f)
		assert "mae" in m and "r2" in m
