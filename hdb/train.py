from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import load_config
from .features import build_preprocessor, load_training_dataframe
from .utils import get_logger, save_json, utc_now_str


logger = get_logger("train")


def train_model() -> Tuple[str, Dict]:
	cfg = load_config()
	df = load_training_dataframe()
	y = df[cfg.training.target].values
	X = df.drop(columns=[cfg.training.target])

	preprocessor = build_preprocessor(df)

	if cfg.training.model_type == "RandomForestRegressor":
		model = RandomForestRegressor(
			n_estimators=cfg.training.n_estimators,
			max_depth=cfg.training.max_depth,
			random_state=cfg.training.random_state,
			n_jobs=-1,
		)
	else:
		raise ValueError("Unsupported model_type")

	pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.training.test_size, random_state=cfg.training.random_state
	)
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	mae = float(mean_absolute_error(y_test, preds))
	r2 = float(r2_score(y_test, preds))
	metrics = {
		"timestamp": utc_now_str(),
		"n_train": int(len(X_train)),
		"n_test": int(len(X_test)),
		"mae": mae,
		"r2": r2,
	}

	# persist
	os.makedirs(cfg.paths.model_dir, exist_ok=True)
	model_path = os.path.join(cfg.paths.model_dir, "rf_pipeline.joblib")
	joblib.dump(pipeline, model_path)
	save_json(metrics, cfg.paths.metrics_path)
	logger.info(f"Saved model to {model_path}; metrics: MAE={mae:.2f}, R2={r2:.3f}")
	return model_path, metrics


if __name__ == "__main__":
	train_model()
